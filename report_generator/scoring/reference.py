"""Import an outside report (e.g. one written in the Claude app) as a reference run for comparison.

The reference is saved like a run, with status "reference", so the compare page can put it next to
a pipeline run and the preference you record becomes a DPO pair. It is never scored for returns.

Usage:
  python -m report_generator.scoring.reference "data/Wayfair (NYSE W) 股票研究報告.docx" --company Wayfair --ticker W \
      --query "幫我生成Wayfair股票研究報告，要包含合理股價，競爭對手，價值鏈，產業分析，總經分析"
Accepts .docx or .md/.txt.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from report_generator.run_log import RUNS_DIR

_HEADINGS = {"Heading1": "# ", "Heading2": "## ", "Heading3": "### ", "Title": "# "}


def _text(fragment: str) -> str:
    return html.unescape("".join(re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", fragment, re.S)))


def docx_to_markdown(path: Path) -> str:
    """Headings, paragraphs, list items and tables from a .docx, as Markdown."""
    xml = zipfile.ZipFile(path).read("word/document.xml").decode()
    out: List[str] = []
    for block in re.findall(r"<w:tbl>.*?</w:tbl>|<w:p[ >].*?</w:p>", xml, re.S):
        if block.startswith("<w:tbl>"):
            rows = [[_text(c).strip() for c in re.findall(r"<w:tc>.*?</w:tc>", r, re.S)]
                    for r in re.findall(r"<w:tr[ >].*?</w:tr>", block, re.S)]
            if rows:
                out += ["", "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
                out += ["| " + " | ".join(r) + " |" for r in rows[1:]] + [""]
            continue
        text = _text(block).strip()
        if not text:
            continue
        style = re.search(r'<w:pStyle w:val="([^"]+)"', block)
        prefix = _HEADINGS.get(style.group(1) if style else "", "- " if "<w:numPr>" in block else "")
        out.append(("\n" if prefix.startswith("#") else "") + prefix + text)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def import_reference(path: Path, company: str, query: str, ticker: str = "", label: str = "reference") -> Path:
    text = docx_to_markdown(path) if path.suffix.lower() == ".docx" else path.read_text()
    now = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{company}-{label}".lower()).strip("-")
    out = RUNS_DIR / f"{now.strftime('%Y-%m-%dT%H-%M-%SZ')}_{slug}.json"
    record: Dict[str, Any] = {
        "started_at": now.isoformat(), "duration_s": 0, "status": "reference",
        "request": {"company_name": company, "ticker": ticker or None, "query": query},
        "failure": None, "result": {"ticker": ticker or None, "report": text},
        "reference": {"source_file": path.name, "label": label},
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    out.with_suffix(".md").write_text(text)
    return out


def reference_runs() -> List[str]:
    ids = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        try:
            if json.loads(p.read_text()).get("status") == "reference":
                ids.append(p.stem)
        except (ValueError, OSError):
            continue
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--company", required=True)
    ap.add_argument("--query", required=True, help="The prompt that produced the report")
    ap.add_argument("--ticker", default="")
    ap.add_argument("--label", default="reference", help="e.g. claude-app")
    a = ap.parse_args()
    print(import_reference(a.path, a.company, a.query, a.ticker, a.label))


if __name__ == "__main__":
    main()
