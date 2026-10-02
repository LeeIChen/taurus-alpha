"""Save every /research run to runs/ as one timestamped JSON file. Local disk only."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("uvicorn.error")

RUNS_DIR = Path(os.environ.get("TAURUS_RUNS_DIR", Path(__file__).resolve().parent.parent / "runs"))


def save_run(
    request: Dict[str, Any],
    started_at: datetime,
    duration_s: float,
    result: Optional[Dict[str, Any]] = None,
    failure: Optional[str] = None,
) -> Optional[Path]:
    """Write one run record. Never raises: a logging problem must not fail the request."""
    record = {
        "started_at": started_at.isoformat(),
        "duration_s": round(duration_s, 1),
        "status": "failed" if failure else "succeeded",
        "request": request,
        "failure": failure,
        "result": result,
    }
    slug = re.sub(r"[^a-z0-9]+", "-", str(request.get("company_name", "unknown")).lower()).strip("-")
    stem = f"{started_at.strftime('%Y-%m-%dT%H-%M-%SZ')}_{slug or 'unknown'}"
    path = RUNS_DIR / f"{stem}.json"
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        n = 2
        while path.exists():  # same company started within the same second
            path = RUNS_DIR / f"{stem}-{n}.json"
            n += 1
        path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    except OSError as exc:
        logger.warning("Could not save run to %s: %s", path, exc)
        return None
    logger.info("Saved run to %s", path)
    return path


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
