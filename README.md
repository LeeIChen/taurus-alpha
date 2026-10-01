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
│   │   └── rag_search.py   # Hybrid Search & Metadata Index Tool
│   └── workflow.py         # LangGraph State Machine Design
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
  -d '{"company_name": "Apple", "query": "Is Apple attractively valued after the latest 10-K?"}'
```

The search index (`app/tools/rag_search.py`) starts empty — load filings into
`default_index` at startup, tagging each chunk with `company`, `doc_type` and
`page_number` metadata. `app/tools/code_executor.py` is not a security
sandbox; run the service in a container before exposing it.
