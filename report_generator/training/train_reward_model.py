"""Train a reward model on your preference pairs with LoRA (the RLHF reward step).

Input: runs/reviews/dpo_pairs.jsonl (human pairs from the compare page; GET /export/dpo writes it)
and optionally runs/reviews/editor_pairs.jsonl (automatic pairs from the editor loop).
Output: a LoRA adapter for a sequence-classification head on an open-weight model. Serve it with
scripts/serve_local_model.py and set TAURUS_REWARD_URL + TAURUS_BEST_OF so the section writers draft
several candidates and keep the one the reward model scores highest (best-of-n).

Needs a CUDA GPU (about 24 GB with --load-in-4bit, 40 GB+ in bf16 for 8k-token memos); it will not
train in reasonable time on a Mac. Install: pip install -r requirements-train.txt

  python report_generator/training/train_reward_model.py --out models/reward-lora
  python report_generator/training/train_reward_model.py --base Qwen/Qwen2.5-7B-Instruct --editor-pairs --epochs 2

Rule of thumb: below about 200 human pairs the model mostly learns the editor's style; check the
held-out accuracy it prints before trusting it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEWS = ROOT / "runs" / "reviews"


def load_pairs(editor_pairs: bool) -> list[dict]:
    rows = []
    files = [REVIEWS / "dpo_pairs.jsonl"] + ([REVIEWS / "editor_pairs.jsonl"] if editor_pairs else [])
    for f in files:
        if f.exists():
            rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    # The reward model reads prompt + text as one sequence (same format serve_local_model.py scores).
    return [{"chosen": f"{r['prompt']}\n\n{r['chosen']}", "rejected": f"{r['prompt']}\n\n{r['rejected']}"}
            for r in rows if r["chosen"].strip() and r["rejected"].strip() and r["chosen"] != r["rejected"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct", help="Open-weight base model (Hugging Face id)")
    ap.add_argument("--out", default=str(ROOT / "models" / "reward-lora"))
    ap.add_argument("--editor-pairs", action="store_true", help="Also train on automatic editor-loop pairs")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--load-in-4bit", action="store_true", help="QLoRA: fits a 7-8B model on a 24 GB GPU")
    a = ap.parse_args()

    pairs = load_pairs(a.editor_pairs)
    if len(pairs) < 4:
        raise SystemExit(f"Only {len(pairs)} usable pairs; record more on /compare first.")
    random.Random(0).shuffle(pairs)
    n_eval = max(1, len(pairs) // 10)
    eval_rows, train_rows = pairs[:n_eval], pairs[n_eval:]
    print(f"{len(train_rows)} training pairs, {len(eval_rows)} held out")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    from trl import RewardConfig, RewardTrainer

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if a.load_in_4bit else None
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=1, torch_dtype=torch.bfloat16,
                                                               quantization_config=quant)
    model.config.pad_token_id = tok.pad_token_id

    trainer = RewardTrainer(
        model=model,
        args=RewardConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
                          per_device_train_batch_size=1, gradient_accumulation_steps=8, max_length=a.max_length,
                          gradient_checkpointing=True, bf16=True, logging_steps=5, eval_strategy="epoch",
                          save_strategy="epoch", report_to=[]),
        train_dataset=Dataset.from_list(train_rows), eval_dataset=Dataset.from_list(eval_rows),
        processing_class=tok,
        peft_config=LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear"),
    )
    trainer.train()
    print("Held-out metrics (accuracy = share of pairs ranked the way you chose):", trainer.evaluate())
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    (Path(a.out) / "taurus.json").write_text(json.dumps({"base": a.base, "kind": "reward", "pairs": len(pairs)}))
    print(f"Saved adapter to {a.out}")


if __name__ == "__main__":
    main()
