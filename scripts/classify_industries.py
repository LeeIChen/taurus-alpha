"""Profile every GICS sub-industry in the screener dataset with Claude (paid).

  python scripts/classify_industries.py --estimate   # how many are missing and what they would cost
  python scripts/classify_industries.py --limit 3    # try a few first and see the measured cost
  python scripts/classify_industries.py              # all missing ones
  python scripts/classify_industries.py --redo "Semiconductors"

Profiles are cached in data/screener/industry_profiles.json; only missing ones are sent.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm import start_meter, summarize_costs  # noqa: E402
from screener import profiles  # noqa: E402
from screener.features import load_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--redo", nargs="*", default=None, help="sub-industry names to classify again")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    data = load_dataset()
    industries = data["industries"]
    companies = {c["ticker"]: c for c in data["companies"]}
    todo = args.redo if args.redo else profiles.missing(industries)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} sub-industries to profile, about ${len(todo) * profiles.EST_USD_PER_INDUSTRY:.2f}")
    if args.estimate or not todo:
        return
    meter = start_meter()
    result = profiles.classify_all(industries, companies, names=todo)
    cost = summarize_costs(meter)
    n = max(result["profiled"], 1)
    print(f"Profiled {result['profiled']}, failed {len(result['failed'])}: {result['failed']}")
    print(f"Cost ${cost['total_usd']:.2f} (${cost['total_usd'] / n:.3f} per industry)")


if __name__ == "__main__":
    main()
