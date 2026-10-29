"""Distill the industry labeler into a small open-weight model with LoRA (supervised fine-tuning).

Teacher labels: Claude's industry profiles and cited 10-K checks; the investor's corrections replace
the teacher's label where they disagree (and are repeated --human-weight times, so they count more).
Served by scripts/serve_local_model.py (/label) with TAURUS_LABELER_URL set, profiles.py tries it
first and only calls Claude when its answer doesn't validate, so re-labeling the universe after a
data refresh costs nothing.

Needs a CUDA GPU for 3B+ models; a 0.5-1.5B base trains on Apple Silicon (MPS) slowly.
Install: pip install -r requirements-train.txt

  python screener/training/train_labeler.py --out models/labeler-lora
  python screener/training/train_labeler.py --base Qwen/Qwen2.5-3B-Instruct --epochs 3
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
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "models" / "labeler-lora"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--human-weight", type=int, default=3, help="copies of each human-corrected example")
    ap.add_argument("--load-in-4bit", action="store_true")
    a = ap.parse_args()

    from screener.training.export import label_examples

    rows = label_examples()
    examples = []
    for r in rows:
        ex = {"prompt": [{"role": "user", "content": r["prompt"]}],
              "completion": [{"role": "assistant", "content": r["completion"]}]}
        examples += [ex] * (a.human_weight if r["human_corrected"] else 1)
    if len(examples) < 20:
        raise SystemExit(f"Only {len(examples)} examples; profile industries first (scripts/classify_industries.py).")
    random.Random(0).shuffle(examples)
    n_eval = max(2, len(examples) // 10)
    print(f"{len(examples) - n_eval} training examples ({sum(bool(r['human_corrected']) for r in rows)} human-corrected), "
          f"{n_eval} held out")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.base)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if a.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, quantization_config=quant)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, per_device_train_batch_size=1,
                       gradient_accumulation_steps=8, max_length=4096, completion_only_loss=True,
                       gradient_checkpointing=True, bf16=torch.cuda.is_available(), logging_steps=5,
                       eval_strategy="epoch", save_strategy="epoch", report_to=[]),
        train_dataset=Dataset.from_list(examples[n_eval:]), eval_dataset=Dataset.from_list(examples[:n_eval]),
        processing_class=tok,
        peft_config=LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear"),
    )
    trainer.train()
    print("Held-out loss:", trainer.evaluate())
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    (Path(a.out) / "taurus.json").write_text(json.dumps({"base": a.base, "kind": "labeler", "examples": len(examples)}))
    print(f"Saved adapter to {a.out}; serve with LABELER_ADAPTER={a.out} uvicorn scripts.serve_local_model:app --port 8100")


if __name__ == "__main__":
    main()
