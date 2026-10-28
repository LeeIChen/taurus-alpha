"""Train a reward model on the investor's screener choices with LoRA (the RLHF reward step).

Input: preference pairs from the shortlist (A-or-B answers, and every thumbs-up against every
thumbs-down in the same session), as company cards: one line with the company's labels and key
figures, plus the session's goals and filters as the prompt.
Output: a LoRA adapter with a scalar head. Serve it with scripts/serve_local_model.py
(REWARD_ADAPTER=...) and set TAURUS_SCREEN_REWARD_URL; the shortlist then blends its score with the
quality score and the Bradley-Terry weights fitted live in screener/preferences.py.

Cards are short, so a 0.5-1.5B base is enough and trains on Apple Silicon (MPS) or any CUDA GPU.
Install: pip install -r requirements-train.txt

  python screener/training/train_pref_reward.py --out models/screen-reward-lora
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "models" / "screen-reward-lora"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    from screener.preferences import export_pairs

    # Same "prompt\n\ntext" format serve_local_model.py scores.
    pairs = [{"chosen": f"{p['prompt']}\n\n{p['chosen']}", "rejected": f"{p['prompt']}\n\n{p['rejected']}"}
             for p in export_pairs() if p["chosen"] != p["rejected"]]
    if len(pairs) < 10:
        raise SystemExit(f"Only {len(pairs)} pairs; rate or compare more companies on the shortlist first.")
    random.Random(0).shuffle(pairs)
    n_eval = max(2, len(pairs) // 10)
    print(f"{len(pairs) - n_eval} training pairs, {n_eval} held out")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardConfig, RewardTrainer

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=1, torch_dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    trainer = RewardTrainer(
        model=model,
        args=RewardConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, per_device_train_batch_size=4,
                          max_length=1024, logging_steps=5, eval_strategy="epoch", save_strategy="epoch", report_to=[]),
        train_dataset=Dataset.from_list(pairs[n_eval:]), eval_dataset=Dataset.from_list(pairs[:n_eval]),
        processing_class=tok,
        peft_config=LoraConfig(task_type="SEQ_CLS", r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear"),
    )
    trainer.train()
    print("Held-out metrics (accuracy = share of pairs ranked the way you chose):", trainer.evaluate())
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    (Path(a.out) / "taurus.json").write_text(json.dumps({"base": a.base, "kind": "screen_reward", "pairs": len(pairs)}))
    print(f"Saved adapter to {a.out}; serve with REWARD_ADAPTER={a.out} uvicorn scripts.serve_local_model:app --port 8101")


if __name__ == "__main__":
    main()
