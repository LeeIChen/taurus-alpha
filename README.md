# taurus-alpha
This is a LangGraph agent system that helps with decisions in investing listed companies.

## 📁 Project Structure

```text
pe-investment-agent/
├── app/
│   ├── __init__.py
│   ├── state.py            # 全局 State 與 Pydantic Schemas
│   ├── agents/
│   │   ├── planner.py      # Planner Agent
│   │   ├── retrieval.py    # Data Retrieval Agent
│   │   ├── analysis.py     # Financial Analysis Agent
│   │   └── writer.py       # Writer Agent
│   ├── tools/
│   │   ├── code_executor.py# Python 沙盒計算工具
│   │   └── rag_search.py   # Hybrid Search & Metadata 檢索工具
│   └── workflow.py         # LangGraph State Machine 編排
├── main.py                 # FastAPI 入口點
├── requirements.txt
└── README.md
```
