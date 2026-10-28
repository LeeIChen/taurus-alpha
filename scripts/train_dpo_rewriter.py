"""Train a memo rewriter with DPO + LoRA: draft + editor notes -> the revision you prefer.

Input: runs/reviews/editor_pairs.jsonl (GET /export/editor-pairs writes it), one row per revision
with the notes it applied. Each pair teaches the model to prefer the revised memo over leaving the
draft as it was. Human pairs from the compare page can be added with --human-pairs, used as
"rewrite the weaker memo toward the stronger one".

Output: a LoRA adapter for a causal LM. Serve it with scripts/serve_local_model.py and set
TAURUS_REWRITER_URL; the reviser then tries it before Claude and keeps its output only if it adds
no number that was not in the draft (app/reward.py: numbers_preserved).

Needs a CUDA GPU (about 24 GB with --load-in-4bit). Install: pip install -r requirements-train.txt

  python scripts/train_dpo_rewriter.py --out models/rewriter-lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEWS = ROOT / "runs" / "reviews"

# Must match what app/agents/editor.py sends: reward.rewrite(f"{REVISE_SYSTEM}\n\nEditor's notes:\n...", memo)
# and how serve_local_model.py builds the user message.
REVISE_SYSTEM = """You are revising an equity research memo to apply the editor's notes. Apply every
note. Keep the rating, price target, every number and every citation as they are, and add no fact
that is not already in the memo or the evidence. Keep the outline and headings. Return the full
revised memo."""


def user_message(instructions: str, memo: str) -> str:
    return f"{instructions}\n\nMemo:\n{memo}"


def load_rows(human_pairs: bool) -> list[dict]:
    rows = []
    f = REVIEWS / "editor_pairs.jsonl"
    for r in (json.loads(x) for x in f.read_text().splitlines() if x.strip()) if f.exists() else []:
        notes = "\n".join(f"- {n}" for n in r["meta"].get("notes") or []) or "- Tighten and remove repetition."
        rows.append({"prompt": [{"role": "user", "content": user_message(f"{REVISE_SYSTEM}\n\nEditor's notes:\n{notes}",
                                                                         r["rejected"])}],
                     "chosen": [{"role": "assistant", "content": r["chosen"]}],
                     "rejected": [{"role": "assistant", "content": r["rejected"]}]})
    h = REVIEWS / "dpo_pairs.jsonl"
    if human_pairs and h.exists():
        for r in (json.loads(x) for x in h.read_text().splitlines() if x.strip()):
            rows.append({"prompt": [{"role": "user", "content": user_message(
                             f"{REVISE_SYSTEM}\n\nEditor's notes:\n- Make it as clear and concise as the best memos.",
                             r["rejected"])}],
                         "chosen": [{"role": "assistant", "content": r["chosen"]}],
                         "rejected": [{"role": "assistant", "content": r["rejected"]}]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "models" / "rewriter-lora"))
    ap.add_argument("--human-pairs", action="store_true")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta: higher stays closer to the base model")
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--load-in-4bit", action="store_true")
    a = ap.parse_args()

    rows = load_rows(a.human_pairs)
    if len(rows) < 4:
        raise SystemExit(f"Only {len(rows)} pairs; run the pipeline a few times (the editor loop creates them).")
    print(f"{len(rows)} DPO pairs")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if a.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, quantization_config=quant)
    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, beta=a.beta,
                       per_device_train_batch_size=1, gradient_accumulation_steps=8, max_length=a.max_length,
                       max_prompt_length=a.max_length // 2, gradient_checkpointing=True, bf16=True,
                       logging_steps=5, save_strategy="epoch", report_to=[]),
        train_dataset=Dataset.from_list(rows),
        processing_class=tok,
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear"),
    )
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    (Path(a.out) / "taurus.json").write_text(json.dumps({"base": a.base, "kind": "rewriter", "pairs": len(rows)}))
    print(f"Saved adapter to {a.out}")


if __name__ == "__main__":
    main()
