"""Serve LoRA adapters over HTTP (see core/reward.py and screener/profiles.py).

  REWARD_ADAPTER=models/reward-lora REWRITER_ADAPTER=models/rewriter-lora \
      uvicorn scripts.serve_local_model:app --port 8100
  REWARD_ADAPTER=models/screen-reward-lora LABELER_ADAPTER=models/labeler-lora \
      uvicorn scripts.serve_local_model:app --port 8101

Report generator (.env): TAURUS_REWARD_URL=http://localhost:8100, TAURUS_BEST_OF=3,
TAURUS_REWRITER_URL=http://localhost:8100.
Screener (.env): TAURUS_SCREEN_REWARD_URL=http://localhost:8101, TAURUS_LABELER_URL=http://localhost:8101.
Any adapter can be left out.

POST /score   {"prompt", "texts": [...]} -> {"scores": [...]}  (higher is preferred)
POST /rewrite {"prompt", "text"}         -> {"text"}
POST /label   {"prompt"}                 -> {"text"}  (label JSON from the distilled labeler)
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="taurus-alpha local models")


def _base(adapter: str) -> str:
    return json.loads((Path(adapter) / "taurus.json").read_text())["base"]


@lru_cache(maxsize=1)
def reward_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    adapter = os.environ.get("REWARD_ADAPTER")
    if not adapter:
        raise HTTPException(404, "REWARD_ADAPTER is not set")
    tok = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForSequenceClassification.from_pretrained(_base(adapter), num_labels=1, torch_dtype=torch.bfloat16,
                                                              device_map="auto")
    base.config.pad_token_id = tok.pad_token_id
    return tok, PeftModel.from_pretrained(base, adapter).eval()


@lru_cache(maxsize=1)
def rewriter_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = os.environ.get("REWRITER_ADAPTER")
    if not adapter:
        raise HTTPException(404, "REWRITER_ADAPTER is not set")
    tok = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForCausalLM.from_pretrained(_base(adapter), torch_dtype=torch.bfloat16, device_map="auto")
    return tok, PeftModel.from_pretrained(base, adapter).eval()


@lru_cache(maxsize=1)
def labeler_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = os.environ.get("LABELER_ADAPTER")
    if not adapter:
        raise HTTPException(404, "LABELER_ADAPTER is not set")
    tok = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForCausalLM.from_pretrained(_base(adapter), torch_dtype=torch.bfloat16, device_map="auto")
    return tok, PeftModel.from_pretrained(base, adapter).eval()


class ScoreIn(BaseModel):
    prompt: str
    texts: List[str]


class RewriteIn(BaseModel):
    prompt: str
    text: str


class LabelIn(BaseModel):
    prompt: str


@app.post("/score")
def score(body: ScoreIn) -> dict:
    import torch

    tok, model = reward_model()
    scores = []
    with torch.no_grad():
        for text in body.texts:  # same "prompt\n\ntext" format as training
            enc = tok(f"{body.prompt}\n\n{text}", return_tensors="pt", truncation=True, max_length=8192).to(model.device)
            scores.append(float(model(**enc).logits[0, 0]))
    return {"scores": scores}


@app.post("/rewrite")
def rewrite(body: RewriteIn) -> dict:
    import torch

    tok, model = rewriter_model()
    messages = [{"role": "user", "content": f"{body.prompt}\n\nMemo:\n{body.text}"}]  # as in train_dpo_rewriter.py
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=8192, do_sample=False)
    return {"text": tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)}


@app.post("/label")
def label(body: LabelIn) -> dict:
    import torch

    tok, model = labeler_model()
    messages = [{"role": "user", "content": body.prompt}]  # as in screener/training/train_labeler.py
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=2048, do_sample=False)
    return {"text": tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)}
