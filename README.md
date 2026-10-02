# taurus-alpha
This is a investment agent system that helps with decisions in investing listed companies.

## 📁 Project Structure

```text
taurus-alpha/
├── app/
│   ├── __init__.py
│   ├── state.py            # Global State 與 Pydantic Schemas
│   ├── llm.py              # Shared Claude client (model, effort, refusal fallback)
│   ├── agents/
│   │   ├── planner.py      # Planner Agent
│   │   ├── retrieval.py    # Data Retrieval Agent
│   │   ├── analysis.py     # Financial Analysis Agent
│   │   ├── writer.py       # Writer Agent
│   │   └── verifier.py     # Faithfulness check on the draft report
│   ├── tools/
│   │   ├── code_executor.py# Python Sandbox Calculation Tool
│   │   ├── valuation.py    # DCF and P/E target price math
│   │   ├── rag_search.py   # Hybrid Search & Metadata Index Tool
│   │   └── embeddings.py   # Local embedding model for the dense half of search
│   └── workflow.py         # LangGraph State Machine Design
├── scripts/
│   └── fetch_filings.py    # Download latest 10-K/10-Q from SEC EDGAR
├── data/filings/           # Page-level JSONL chunks loaded into the search index
├── runs/                   # One JSON record per /research run (gitignored)
├── main.py                 # FastAPI Entrypoint
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

Requires **Python 3.10+**.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # see .env.example for optional overrides
uvicorn main:app --reload
```

```bash
curl -X POST localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"company_name": "Apple", "query": "Is Apple attractively valued after the latest 10-K?", "current_price": 230}'
```

## 📄 Filing Data

`scripts/fetch_filings.py` downloads the latest 10-K and 10-Q for the Magnificent 7
(Apple, Microsoft, Alphabet, Amazon, Meta, Nvidia, Tesla) from SEC EDGAR and writes
one JSONL file per filing to `data/filings/<TICKER>/`, one chunk per page, with
`company`, `ticker`, `doc_type`, `period_of_report`, `filing_date`, `page_number`
and `source_url` metadata. `data/filings/manifest.json` lists what was fetched.

```bash
export SEC_USER_AGENT="taurus-alpha you@example.com"   # SEC requires a contact email
python scripts/fetch_filings.py
```

The API loads every `*.jsonl` under `data/filings` (or `TAURUS_FILINGS_DIR`) at
startup. Company names are matched case-insensitively and by ticker, plus a few
aliases (e.g. "Google" → Alphabet, "Facebook" → Meta).

Search is hybrid: BM25 keyword ranking and dense embeddings
(`BAAI/bge-small-en-v1.5`, run locally through `fastembed`, no API key) merged
with Reciprocal Rank Fusion. Each page is embedded in ~250-word windows and
scored by its best window. The first startup embeds every filing (about
5 minutes on a laptop CPU) and caches the vectors in `data/filings/.embeddings/`;
later startups only embed new text. Set `TAURUS_EMBEDDINGS=off` for BM25 only.

## 🗂️ Run History

Every `/research` call is saved to `runs/<UTC timestamp>_<company>.json`. Each file holds
the request, duration, status, and either the full result (plan, metrics, citations,
valuation, memo, faithfulness score, errors) or the failure message. Saving is local
disk only, with no extra API calls. Set `TAURUS_RUNS_DIR` to save somewhere else. `runs/`
is gitignored.

## 💰 Target Price

The analysis agent values the company two ways and blends them:

- **DCF:** 5 years of projected free cash flow plus a terminal value, discounted at
  the chosen WACC, plus net cash, divided by diluted shares.
- **P/E:** forward EPS × target multiple.

Claude picks the inputs: reported figures cited from the filings, and projections
(growth, WACC, multiple, blend weight) justified in a written rationale.
`app/tools/valuation.py` then does the math, including a WACC × terminal-growth
sensitivity grid. The filings contain no share prices, so pass the optional
`current_price` to get upside/downside.

## 🔎 Peer Filings

When a research task names another covered company (for example "hyperscaler capex at
Microsoft, Alphabet, Amazon and Meta"), retrieval also searches that company's filings.
The writer sees every retrieved page, so it can use peer evidence and doesn't report
gaps the documents already answer.

`app/tools/code_executor.py` is not a security
sandbox; run the service in a container before exposing it.
