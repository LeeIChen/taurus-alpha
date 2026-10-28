"""Training data for the screener's LoRA models.

label_examples(): prompt -> label JSON pairs for the labeler (train_labeler.py). Industry examples
come from the cached Claude profiles with the investor's corrections applied (a correction replaces
the model's label, so the labeler learns the investor's view); company examples come from the cited
10-K deep dives. The prompt format matches what profiles._local_label sends.

Preference pairs for the reward model come from preferences.export_pairs().
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from screener import evidence, profiles

TASK_INDUSTRY = "industry_profile"
TASK_COMPANY = "company_check"


def label_examples() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    data = profiles._load(profiles.PROFILES_PATH)
    ovr = profiles.overrides()
    fields = list(profiles.IndustryProfile.model_fields)
    for name, p in data.items():
        if p.get("source") != "claude" or not p.get("prompt"):
            continue  # only teacher labels (not the labeler's own output)
        label = {k: p.get(k) for k in fields}
        corrected = []
        for field, o in ovr.get(name, {}).items():
            if field in label:
                label[field] = o["value"]
                corrected.append(field)
        rows.append({"task": TASK_INDUSTRY, "name": name, "human_corrected": corrected,
                     "prompt": f"{profiles.SYSTEM}\n\n{p['prompt']}",
                     "completion": json.dumps(label, ensure_ascii=False)})
    fields = list(evidence.CompanyCheck.model_fields)
    for path in sorted(evidence.DEEP_DIR.glob("*.json")) if evidence.DEEP_DIR.exists() else []:
        d = json.loads(path.read_text())
        if not d.get("prompt"):
            continue
        rows.append({"task": TASK_COMPANY, "name": d.get("ticker"), "human_corrected": [],
                     "prompt": f"{evidence.SYSTEM}\n\n{d['prompt']}",
                     "completion": json.dumps({k: d.get(k) for k in fields}, ensure_ascii=False)})
    return rows
