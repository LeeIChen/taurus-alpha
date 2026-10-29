# taurus-alpha

An investment research system with two parts:

- **Screener** (`screener/`): an interactive wizard that narrows the S&P 1500 to fundamentally strong
  companies, one decision at a time: growth or value, size, sensitivity to the cycle and to rates,
  market structure, bargaining power over suppliers and customers, and the growth of the addressable
  market. It learns the investor's preferences from their choices.
- **Report generator** (`report_generator/`): a full research report on one company (fair value,
  competitors, value chain, industry and macro), callable from the screener's shortlist.

Both run on LangGraph, use retrieval over SEC filings (RAG), and train small open-weight models with
LoRA on the investor's feedback (RLHF).

## 📁 Project Structure

```text
taurus-alpha/
├── main.py                    # FastAPI entrypoint: mounts both routers
├── core/                      # Shared by both parts
│   ├── llm.py                 # Claude client: models, effort, caching, cost meter, web research
│   ├── reward.py              # Client for local LoRA models (reward scores, rewriter)
│   └── tools/
│       ├── sec.py             # SEC EDGAR: ticker lookup, filings, XBRL company facts
│       ├── fred.py            # Macro series from FRED (no API key)
│       ├── prices.py          # Daily closes (Yahoo Finance chart endpoint)
│       ├── peer_comps.py      # Margins, FCF and multiples from XBRL facts and closes
│       ├── rag_search.py      # Hybrid BM25 + embedding search over filing pages
│       └── embeddings.py      # Local embedding model for the dense half of search
├── screener/
│   ├── universe.py            # S&P 500/400/600 constituents with GICS sub-industry
│   ├── fundamentals.py        # XBRL frames for the whole universe; margins, returns, F-score, elasticity
│   ├── market.py              # Monthly prices; market, rate and leading-index betas
│   ├── macro.py               # Composite leading index and the current cycle and rate regime
│   ├── features.py            # Dataset build: company labels and industry aggregates
│   ├── profiles.py            # Claude (or the LoRA labeler) classifies each sub-industry
│   ├── hedge.py               # Hedge-fund questions: reverse DCF, ROIC vs WACC, revenue quality, crowding
│   ├── evidence.py            # 10-K deep dive on shortlisted companies (RAG, cited)
│   ├── hedge_check.py         # Hedge-fund check: contracts, estimates, catalysts, short interest (RAG + web)
│   ├── preferences.py         # Feedback store, Bradley-Terry preference model, reward-model blend
│   ├── graph.py               # LangGraph wizard with interrupt() steps and a SQLite checkpointer
│   ├── api.py                 # /screener page and /screen endpoints
│   ├── static/screener.html   # The wizard (Traditional Chinese)
│   └── training/              # LoRA labeler (SFT) and preference reward model; data export
├── report_generator/
│   ├── agents/                # Planner, ingest, market data, web research, retrieval, analysis,
│   │                          # valuation, thesis, sections, editor, verifier
│   ├── tools/                 # Valuation math, Python calculation sandbox
│   ├── scoring/               # Outcome scoring, review and compare pages, references, DPO export
│   ├── training/              # LoRA reward model and DPO rewriter
│   ├── state.py, workflow.py  # State schema and LangGraph pipeline
│   ├── run_log.py             # Run records in runs/
│   └── api.py                 # /research endpoints
├── scripts/
│   ├── build_screener_data.py # Free data build for the screener (SEC, Yahoo, FRED)
│   ├── classify_industries.py # Claude industry profiles (paid; estimates first)
│   ├── fetch_filings.py       # Download latest 10-K/10-Q from SEC EDGAR
│   └── serve_local_model.py   # HTTP server for the LoRA adapters
├── tests/                     # pytest, no network or Claude calls
├── data/                      # Filings, caches, screener dataset and feedback (gitignored)
├── runs/                      # Report run records and reviews (gitignored)
├── requirements.txt
└── requirements-train.txt     # Optional training dependencies
```

## 🚀 Getting Started

Requires **Python 3.10+**.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
export SEC_USER_AGENT="your-name you@example.com"   # SEC requires a contact

python scripts/build_screener_data.py               # free; 10-20 minutes the first time
python scripts/classify_industries.py --estimate     # then without --estimate (about $2 for 153 industries)
uvicorn main:app --reload                            # open http://localhost:8000/screener
```

A single report without the screener:

```bash
curl -X POST localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"company_name": "Moderna", "ticker": "MRNA", "query": "Price target from DCF and P/E, value by program, buy or sell?"}'
```

Tests (no network or Claude calls): `PYTHONPATH=. pytest -q tests`

## 🧭 Screener

### Data (no model calls)

- **Universe:** S&P 500 + MidCap 400 + SmallCap 600 from Wikipedia's constituent tables, about
  1,500 companies in 153 GICS sub-industries, one row per company (dual share classes merged).
- **Financials:** SEC XBRL frames. One request returns every filer's value for one line item and
  one calendar period, so about 400 requests cover the universe for 2015 to last year. Banks without a
  revenue total use net interest income plus noninterest income.
- **Prices:** 10 years of monthly adjusted closes from Yahoo Finance; market cap uses the cover-page
  share count.
- **Macro:** ISM PMI isn't on FRED and the OECD leading indicator stopped in 2024, so a composite
  leading index is built from the Conference Board LEI components that FRED carries (manufacturing
  hours, jobless claims, building permits, capital-goods orders, the 10y-3m spread, sentiment).

### The dimensions

| Step | Label | How it is decided | Level |
|---|---|---|---|
| Growth or value | 成長 / 均衡 / 價值 | Growth percentile (3/5-year revenue CAGR, latest-quarter growth, 3-year earnings CAGR) minus value percentile (earnings, book, sales, FCF and dividend yields); a 15-point gap decides. The industry's nature is also judged by the model | Company or industry |
| Size | 大型 / 中型 / 小型 | Market cap ≥ $20B, $5-20B, < $5B | Company |
| Cycle | 高 / 中 / 低 | Mean percentile of market beta, beta to the leading index (market held fixed) and revenue elasticity to industrial production; tertiles | Company or industry |
| Rates | 升息受害 / 不敏感 / 升息受惠 | Monthly return per 1 pp rise in the 10-year yield, market held fixed; \|t\| ≥ 1.65 | Company or industry |
| Market structure | 獨占 / 寡占 / 壟斷性競爭 / 接近完全競爭 | Model judgment of the global market, given HHI and top-4 share among index members | Industry |
| Bargaining power | 強 / 中 / 弱, over suppliers and over customers | Model judgment given gross margin level and stability, receivable and payable days | Industry; checked per company against its 10-K |
| TAM growth | 高成長 / 中度成長 / 低成長 / 萎縮 | Model estimate of the 5-year market CAGR (≥ 10%, 4-10%, 0-4%, < 0), shown next to members' past revenue growth; one industry can be checked on the web | Industry |

Then a **quality** bar: the sector-relative percentile of ROIC, ROE, gross and operating margin,
FCF margin, Piotroski F-score, net debt/EBITDA, interest coverage and margin stability (banks and
insurers use ROE, ROA, equity/assets, F-score and earnings stability instead), plus optional
F-score, leverage and positive-FCF thresholds.

### The hedge-fund questions (`screener/hedge.py`)

After the quality bar, three steps ask what a fundamental long/short fund asks before buying a good
business. All labels are computed from the dataset; each step lists the top companies with the numbers
behind them.

| Step | Label | How it is decided |
|---|---|---|
| Who holds the bottleneck? | 關鍵瓶頸 / 部分瓶頸 / 非瓶頸 (industry); 定價權 增強 / 持平 / 減弱 (company) | The model judges whether customers queue for the industry's capacity, equipment, permits or grid access, and notes its place in the AI build-out. Per company: gross margin vs its 5-year median, and backlog (RPO) growth vs revenue growth |
| How good is the revenue? | 高 / 中 / 低 | Percentiles of backlog in years of revenue (`RevenueRemainingPerformanceObligation`, about 340 filers), revenue-growth volatility, down years and gross-margin stability; tertiles |
| How much is priced in? | 預期偏低 / 大致反映 / 預期偏高 / 極度樂觀 | Reverse DCF: the 10-year cash-flow growth (then 2.5% forever) that today's EV implies, vs the mean 3- and 5-year revenue CAGR (gaps of −3, +3 and +10 points). Companies spending over 1.5× D&A use operating cash flow minus D&A, since reported FCF is net of the investment behind the growth; banks and insurers use the ROE implied by price-to-book |
| Does it pay at today's rates? | 通過 / 勉強 / 不通過 | WACC from the current 10-year yield + Blume-adjusted beta × 5%, and interest paid / debt. Pass: ROIC (ROE for financials) ≥ WACC + 2 points, interest cover ≥ 4 and no debt-funded buybacks and dividends; fail: below WACC or cover < 2 |
| Crowding | 高 / 中 / 低 | Percentiles of 12-month momentum, valuation premium to the sector (EV/EBITDA, P/E) and the reverse-DCF gap; tertiles. The drawdown from the 12-month high is shown too |
| Catalysts | 基本面 加速 / 持平 / 放緩 | Latest-quarter revenue growth vs the 3-year CAGR (±3 points) |

What the numbers can't show (contract share and length, spot-price exposure, estimate revisions, short
interest, dated catalysts, what would prove the thesis wrong) comes from the **hedge-fund check** on the
shortlist (`screener/hedge_check.py`): the 10-K's backlog, contract and customer pages (RAG) plus web
search, about $0.25 per company (measured $0.15 with 5 searches; now 8), cached 7 days.

### How the techniques are used

- **LangGraph:** `screener/graph.py` is a state machine whose steps pause with `interrupt()` and
  resume with the investor's answer (`Command(resume=...)`); answers can go back or jump to any
  visited step. Sessions are checkpointed in SQLite, so they survive a restart.
- **RAG:** a deep dive downloads a shortlisted company's latest 10-K into the shared hybrid index
  and retrieves its competition, customer, supplier, pricing and market passages; Claude labels the
  company's moat and bargaining power with each claim cited to a filing page.
- **RLHF:** on the shortlist the investor rates companies and answers A-or-B questions. A
  Bradley-Terry model on standardized features is refit on every answer and reranks the list (its
  weight grows with the number of pairs); its strongest weights are shown as "你偏好…".
  Corrections to any industry label win over the model from then on.
- **LoRA:** `screener/training/train_pref_reward.py` trains a reward model on the same pairs (company
  cards as text; a 0.5B base trains on a Mac), blended into the ranking with
  `TAURUS_SCREEN_REWARD_URL`. `train_labeler.py` distills Claude's industry and 10-K labels, with
  the investor's corrections taking precedence, into a small model; with `TAURUS_LABELER_URL` set
  it labels industries first and Claude is only called when its answer doesn't validate.

### Cost

Only these steps call Claude (Sonnet 5.5): profiling the industries (once, about $0.013 each), the
optional one-line goals at the start (about $0.01), a 10-K deep dive (about $0.08 per company,
cached 30 days), a hedge-fund check (about $0.25 per company, cached 7 days), a TAM web check (about
$0.15) and the full report (about $3.60). The page asks
before each paid step and shows the session's spend.

# 📝 Report Generator

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
