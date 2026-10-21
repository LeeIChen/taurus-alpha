# taurus-alpha
This is a investment agent system that helps with decisions in investing listed companies.

## 📁 Project Structure

```text
taurus-alpha/
├── app/
│   ├── __init__.py
│   ├── state.py            # Global State 與 Pydantic Schemas
│   ├── llm.py              # Shared Claude client (model, effort, refusal fallback)
│   ├── scoring/            # Outcome scoring, human review pages, DPO export
│   ├── agents/
│   │   ├── planner.py      # Planner Agent
│   │   ├── ingest.py       # Company intake: SEC lookup, filings, reported financials
│   │   ├── web_research.py # Web Research Agent (cited web search and fetch)
│   │   ├── retrieval.py    # Data Retrieval Agent
│   │   ├── analysis.py     # Financial Analysis Agent
│   │   ├── valuation.py    # Valuation Agent (company DCF or pipeline sum of the parts, plus P/E)
│   │   ├── writer.py       # Investment thesis, rating and memo
│   │   └── verifier.py     # Faithfulness check on the draft report
│   ├── tools/
│   │   ├── code_executor.py# Python Sandbox Calculation Tool
│   │   ├── valuation.py    # DCF, P/E and risk-adjusted NPV math
│   │   ├── sec.py          # SEC EDGAR: ticker lookup, filings, XBRL financials
│   │   ├── rag_search.py   # Hybrid Search & Metadata Index Tool
│   │   └── embeddings.py   # Local embedding model for the dense half of search
│   └── workflow.py         # LangGraph State Machine Design
├── scripts/
│   └── fetch_filings.py    # Download latest 10-K/10-Q from SEC EDGAR
├── data/filings/           # Page-level JSONL chunks loaded into the search index
├── runs/                   # Run records, scores and human reviews (gitignored)
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
  -d '{"company_name": "Moderna", "ticker": "MRNA", "query": "Price target from DCF and P/E, value by program, buy or sell?"}'
```

## 🔁 Research Flow

`POST /research` runs one LangGraph pass over `InvestmentAgentState`:

1. **Planner** breaks the question into research tasks and metrics.
2. **Ingest** resolves the company at SEC (any listed US filer, by ticker or name), downloads its
   latest 10-K and 10-Q if they aren't cached, adds them to search, and reads key reported
   financials from XBRL. Filings become sources `F1`, `F2`, ...
3. **Web research** uses Claude's server-side web search and fetch to cover recent developments,
   products and pipeline, success-rate benchmarks, market size, financing, catalysts, the share
   price and sell-side targets. Every cited fact keeps its URL as source `S1`, `S2`, ...
4. **Retrieval** searches the target's filings, plus any covered peer named in a task.
5. **Analysis** computes metrics by running code, citing filings, web sources or XBRL.
6. **Valuation** picks a method: a company-level DCF, or for pipeline companies a sum of the
   parts with one risk-adjusted NPV per product or program (probability of success from cited
   benchmarks). Both add a P/E value on risk-adjusted earnings; `app/tools/valuation.py` does the math.
7. **Writer** sets the rating (Buy / Hold / Sell), price target, thesis, risks and dated catalysts,
   then writes the memo with a numbered source list.
8. **Verifier** scores how well the memo is supported by the evidence.

Each run is saved to `runs/` as JSON (everything above) and Markdown (tasks, thesis, catalysts,
valuation table, memo and all sources). A deep run uses up to 12 web searches and 6 page fetches.
Web search must be enabled for your organization in the Claude Console.

**Prompt caching.** The steps after retrieval (analysis, valuation, thesis, memo, verifier) read
the same evidence block: SEC financials, sources, research notes and filing pages. They share one
system prompt and put the evidence first, with each step's instructions and JSON schema after it
(`app/agents/schemas.py`), so that block is a byte-identical cached prefix: the first step writes it
(5-minute TTL, restarted by every read) and the other four read it at the cache rate (5% of the
input price on Opus 5.5). Web research turns on automatic
caching so each search iteration re-reads the growing context from cache. Every Claude call logs
its cache reads and writes.

## 📄 Filing Data

`scripts/fetch_filings.py` downloads the latest 10-K and 10-Q for the Magnificent 7
(Apple, Microsoft, Alphabet, Amazon, Meta, Nvidia, Tesla) from SEC EDGAR and writes
one JSONL file per filing to `data/filings/<TICKER>/`, one chunk per page, with
`company`, `ticker`, `doc_type`, `period_of_report`, `filing_date`, `page_number`
and `source_url` metadata. `data/filings/manifest.json` lists what was fetched.

```bash
export SEC_USER_AGENT="taurus-alpha you@example.com"   # SEC requires a contact email
python scripts/fetch_filings.py            # the Magnificent 7
python scripts/fetch_filings.py MRNA PFE   # any tickers (the API also fetches on demand)
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

## 📏 Scoring and Human Review

Every saved run can be checked against what actually happened, and reviewed by a person.

**Automatic scoring** (`python -m app.scoring.score`, or `GET /scores`): the price at the run,
return and excess return vs SPY at 30, 90, 180 and 365 days, whether the rating's direction was
right (Buy beat SPY, Sell lagged it, Hold stayed within ±10%), and the 12-month target error.
Results go to `runs/scores.json`. Prices come from Yahoo Finance's public chart endpoint, cached in
`data/prices/`; add rows to `data/prices/manual.csv` (`ticker,date,close`) to fill gaps.

**Human review** (start the API, then open `http://localhost:8000/review`):
- Score each section 1–5, list factual errors, and give an overall score.
- Correct a program's probability of success or peak sales, and record its outcome once known.
  Recorded outcomes feed a Brier score that shows whether the model's probabilities are calibrated.
- Record what happened for each catalyst; ones whose date has passed are highlighted.
- `/compare` shows two runs side by side to record which is better and why.

Reviews are plain JSON in `runs/reviews/`. `GET /export/dpo` writes
`runs/reviews/dpo_pairs.jsonl` with `{prompt, chosen, rejected}` rows for preference training
(DPO, a simpler form of RLHF) on an open-weight model with LoRA. Claude can't be trained this way;
for Claude the same feedback guides prompt changes and picking the best of several runs.

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
