# taurus-alpha
This is a investment agent system that helps with decisions in investing listed companies.

## 📁 Project Structure

```text
taurus-alpha/
├── app/
│   ├── __init__.py
│   ├── state.py            # Global State 與 Pydantic Schemas
│   ├── agents/
│   │   ├── planner.py      # Planner Agent
│   │   ├── retrieval.py    # Data Retrieval Agent
│   │   ├── analysis.py     # Financial Analysis Agent
│   │   └── writer.py       # Writer Agent
│   ├── tools/
│   │   ├── code_executor.py# Python Sandbox Calculation Tool
│   │   └── rag_search.py   # Hybrid Search & Metadata Index Tool
│   └── workflow.py         # LangGraph State Machine Design
├── main.py                 # FastAPI Entrypoint
├── requirements.txt
└── README.md
```
