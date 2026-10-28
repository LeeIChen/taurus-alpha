"""Build the screener dataset from free sources (no model calls, no cost).

  python scripts/build_screener_data.py            # S&P 1500 universe, SEC XBRL frames, prices, FRED
  python scripts/build_screener_data.py --refresh  # re-read the constituent lists too

Needs SEC_USER_AGENT in the environment (a name and contact email, as SEC requires).
First run: about 300 SEC requests and 1,500 price downloads (10-20 minutes); later runs use caches.
Then classify industries with Claude: python scripts/classify_industries.py (paid; shows an estimate first).
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screener.features import build_dataset  # noqa: E402
from screener.universe import load_universe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download the S&P 400/500/600 lists")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.refresh:
        load_universe(refresh=True)
    build_dataset(progress=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
