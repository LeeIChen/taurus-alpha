# taurus-alpha
This is a investment agent system that helps with decisions in investing listed companies.

## 📁 Project Structure

```text
taurus-alpha/
├── app/
│   ├── __init__.py
│   ├── state.py            # Global State 與 Pydantic Schemas
│   ├── llm.py              # Shared Claude client (model, effort, refusal fallback)
│   ├── reward.py           # Optional local LoRA reward model and rewriter (best-of-n, number guard)
│   ├── scoring/            # Outcome scoring, human review pages, references, DPO export
│   ├── agents/
│   │   ├── planner.py      # Planner Agent
│   │   ├── ingest.py       # Company intake: SEC lookup, filings, reported financials
│   │   ├── market_data.py  # Peer table, financial history and macro series (no model calls)
│   │   ├── web_research.py # Web Research Agent (cited web search and fetch)
│   │   ├── retrieval.py    # Data Retrieval Agent
│   │   ├── analysis.py     # Financial Analysis Agent
│   │   ├── valuation.py    # Valuation Agent (company DCF or pipeline sum of the parts, plus P/E)
│   │   ├── writer.py       # Investment thesis and rating
│   │   ├── sections.py     # Memo sections: per-section retrieval, parallel writers, assembly
│   │   ├── editor.py       # Editor review and revision loop for readability
│   │   └── verifier.py     # Faithfulness check on the draft report
│   ├── tools/
│   │   ├── code_executor.py# Python Sandbox Calculation Tool
│   │   ├── valuation.py    # DCF, P/E and risk-adjusted NPV math
│   │   ├── sec.py          # SEC EDGAR: ticker lookup, filings, XBRL financials
│   │   ├── peer_comps.py   # Margins, FCF and multiples from XBRL facts and closing prices
│   │   ├── fred.py         # Macro series from FRED (no API key)
│   │   ├── rag_search.py   # Hybrid Search & Metadata Index Tool
│   │   └── embeddings.py   # Local embedding model for the dense half of search
│   └── workflow.py         # LangGraph State Machine Design
├── scripts/
│   ├── fetch_filings.py    # Download latest 10-K/10-Q from SEC EDGAR
│   ├── train_reward_model.py  # LoRA reward model on preference pairs (GPU)
│   ├── train_dpo_rewriter.py  # LoRA + DPO memo rewriter (GPU)
│   └── serve_local_model.py   # HTTP server for both adapters
├── data/filings/           # Page-level JSONL chunks loaded into the search index
├── runs/                   # Run records, scores and human reviews (gitignored)
├── main.py                 # FastAPI Entrypoint
├── requirements.txt
├── requirements-train.txt  # Optional training dependencies
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

1. **Planner** breaks the question into research tasks and metrics, and names the closest listed
   peers and the FRED series for the sector's macro drivers.
2. **Ingest** resolves the company at SEC (any listed US filer, by ticker or name), downloads its
   latest 10-K and 10-Q if they aren't cached, adds them to search, and reads key reported
   financials from XBRL. Filings become sources `F1`, `F2`, ...
3. **Market data** computes, without the model: a peer table (TTM revenue, growth, margins,
   EBITDA+SBC, FCF, net debt, market cap, EV/Sales, EV/EBITDA, P/E, FCF yield) from SEC XBRL facts
   and the latest close, the target's multi-year financial history, and a macro table from FRED
   (fed funds, 10-year yield, CPI, unemployment, sentiment, real PCE plus the planner's series).
   It also downloads the peers' filings for search. These are sources `X1` and `M1`.
4. **Web research** uses Claude's server-side web search and fetch to cover recent developments,
   products and pipeline, success-rate benchmarks, market size, financing, catalysts, the share
   price and sell-side targets. Every cited fact keeps its URL as source `S1`, `S2`, ...
5. **Retrieval** searches the target's filings, plus any covered peer named in a task.
6. **Analysis** computes metrics by running code, citing filings, web sources or XBRL.
7. **Valuation** picks a method: a company-level DCF, or for pipeline companies a sum of the
   parts with one risk-adjusted NPV per product or program (probability of success from cited
   benchmarks). Both add a P/E value on risk-adjusted earnings; `report_generator/tools/valuation.py` does the math.
8. **Thesis** sets the rating (Buy / Hold / Sell), price target, thesis, risks and dated catalysts.
9. **Sections** fan out with LangGraph `Send`: eight writers run in parallel, one per section
   (summary, company and latest results, industry with five forces, value chain, competition,
   macro, valuation, catalysts and risks). Each retrieves its own filing pages (and peers' pages
   for industry and competition) and research-note paragraphs for its own queries, on top of a
   compact core (thesis, valuation, metrics, the computed tables). **Assemble** adds the title
   block, numbered headings and the cited-source list in code (`report_generator/agents/sections.py`). A question
   written in Chinese gets the memo in Traditional Chinese.
10. **Editor** reviews the memo for readability: automatic checks (outline, length, process talk
   such as "not retrieved") plus a rubric (conclusion first, no repetition, prose over fragments).
   While it has notes, the **reviser** rewrites the memo and the editor looks again, at most twice
   (`report_generator/agents/editor.py`). The editor never judges facts.
11. **Verifier** scores how well the final memo is supported by the evidence, including the pages
   only the section writers retrieved.

Each run is saved to `runs/` as JSON (everything above, including the drafts before each revision
in `draft_versions` and the notes each revision applied in `edit_history`) and Markdown: the memo as
the reader sees it, then an appendix with the faithfulness score, open editor notes, verifier
warnings and the research plan. A deep run uses up to 8 web searches and 4 page fetches.
Web search must be enabled for your organization in the Claude Console.

**Prompt caching.** The steps after retrieval (analysis, valuation, thesis, verifier) read
the same evidence block: SEC financials, the computed tables, sources, research notes and filing
pages (at most 32 target and 8 peer pages, about 50k tokens). They share one
system prompt and put the evidence first, with each step's instructions and JSON schema after it
(`report_generator/agents/schemas.py`), so that block is a byte-identical cached prefix: the first step writes it
(5-minute TTL, restarted by every read) and the others read it at the cache rate (5% of the
input price on Opus 5.5).

**Cost per run.** The section writers, editor and reviser don't read that cache, so they run on a
cheaper writer model (`TAURUS_WRITER_MODEL`, default `claude-sonnet-5-5`, at `TAURUS_WRITER_EFFORT`
`medium`) without affecting it; the cached steps stay on `TAURUS_MODEL` (Opus 5.5, effort high), since
a cache only serves the same model and effort. Web research skips what the market-data step already
computed. Every call's tokens and USD cost are recorded per step in the run record (`result.cost`,
or `cost` for a failed run), in the Markdown appendix, and in the log line `Run cost $...`. Web research turns on automatic
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

**Automatic scoring** (`python -m report_generator.scoring.score`, or `GET /scores`): the price at the run,
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
(DPO, a simpler form of RLHF) on an open-weight model with LoRA. `GET /export/editor-pairs` writes
`runs/reviews/editor_pairs.jsonl`, one automatic pair per editor revision (revised memo over draft,
with the notes applied); it is AI feedback, kept separate from human pairs.

**Reference reports.** To compare against a report written elsewhere (for example in the Claude
app), import it as a reference run; it then appears on `/compare`:

```bash
python -m report_generator.scoring.reference "data/Wayfair (NYSE W) 股票研究報告.docx" --company Wayfair --ticker W \
    --label claude-app --query "幫我生成Wayfair股票研究報告，要包含合理股價，競爭對手，價值鏈，產業分析，總經分析"
```

## 🧪 RLHF with LoRA (optional, needs a GPU)

Claude can't be fine-tuned on this data, so the preferences train a small open-weight model that
improves Claude's output instead (`pip install -r requirements-train.txt`, CUDA GPU):

1. **Reward model** (`report_generator/training/train_reward_model.py`): LoRA on a 7-8B model with a scalar head,
   trained on your `/compare` pairs (plus `--editor-pairs` to warm up). It prints held-out pairwise
   accuracy; aim for a few hundred human pairs before trusting it.
2. **Best-of-n**: serve it (`uvicorn scripts.serve_local_model:app --port 8100` with
   `REWARD_ADAPTER=models/reward-lora`) and set `TAURUS_REWARD_URL=http://localhost:8100` and
   `TAURUS_BEST_OF=3`. Each section writer then drafts three candidates and keeps the top-scored
   one; all candidates and scores are saved in `section_candidates`. This multiplies section cost by n.
3. **Rewriter** (`report_generator/training/train_dpo_rewriter.py`): DPO + LoRA on the editor pairs, learning
   draft + notes -> revision. With `TAURUS_REWRITER_URL` set, the reviser tries it first and keeps
   its output only if it adds no number that was not in the draft; otherwise Claude revises.

Facts are never left to the local models: the verifier still checks the final memo.

## 💰 Target Price

The analysis agent values the company two ways and blends them:

- **DCF:** 5 years of projected free cash flow plus a terminal value, discounted at
  the chosen WACC, plus net cash, divided by diluted shares.
- **P/E:** forward EPS × target multiple.

Claude picks the inputs: reported figures cited from the filings, and projections
(growth, WACC, multiple, blend weight) justified in a written rationale.
`report_generator/tools/valuation.py` then does the math, including a WACC × terminal-growth
sensitivity grid. The filings contain no share prices, so pass the optional
`current_price` to get upside/downside.

## 🔎 Peer Filings

When a research task names another covered company (for example "hyperscaler capex at
Microsoft, Alphabet, Amazon and Meta"), retrieval also searches that company's filings. The
planner's peer tickers are downloaded automatically in the market-data step, and the industry and
competition sections search each peer's filings directly.

`report_generator/tools/code_executor.py` is not a security
sandbox; run the service in a container before exposing it.
