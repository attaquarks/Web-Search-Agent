# AI Agent Web Search and Research Platform

This repository implements a multi-agent research platform for comparing AI agent reasoning patterns. It includes backend agent implementations, benchmark data, stored evaluation results, and an API layer that can support a web-based playground or dashboard.

## Overview

The project explores how different agent architectures perform on search and research tasks. Agents can use web-search tools, local memory, retrieval, and planning loops, then produce answers with traceable reasoning behavior and measurable benchmark results.

## Supported Agent Patterns

| Agent | Description |
|---|---|
| One-Shot | Direct answer generation for simpler queries |
| Simple RAG | Retrieval-augmented answering over local/contextual data |
| ReAct | Iterative reasoning plus tool use |
| Plan-Execute | Creates a plan, then executes steps sequentially |
| Plan-Execute + Memory | Uses long-term memory retrieval and persistence during planning/execution |

## Features

- Multiple agent implementations behind a shared structure.
- Web-search and retrieval tool integration.
- JSON-backed memory store for experimentation.
- Benchmark runner for comparing answer quality and efficiency.
- Stored result files for each agent type.
- FastAPI bridge for serving agent interactions through an API.
- Frontend-ready structure for chat/playground and evaluation dashboards.

## Tech Stack

- Python
- FastAPI / Uvicorn
- LangChain
- Tavily search
- FAISS / ChromaDB
- sentence-transformers
- pandas, NumPy, Pydantic

## Repository Structure

```text
api_server.py                 FastAPI bridge for agent interaction
src/
  main.py                     CLI / orchestration entry point
  benchmark.py                Evaluation runner
  memory.py                   Long-term memory helpers
  tools.py                    Search/retrieval tools
  agents/
    one_shot.py
    simple_rag.py
    react_agent.py
    plan_execute.py
    plan_execute_memory.py
data/
  test_questions.json         Benchmark questions
  memory_store.json           Local memory store
results/
  comparison_results.json     Aggregated benchmark comparison
  results_*.json              Per-agent evaluation outputs
frontend/                     Optional web interface if present in full checkout
```

## Installation

```bash
git clone https://github.com/attaquarks/Web-Search-Agent.git
cd Web-Search-Agent
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with the required keys for the providers you use:

```env
OPENROUTER_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_key
```

Do not commit real API keys.

## Run the API

```bash
python api_server.py
```

Default API:

```text
http://localhost:8000
```

## Run Benchmarks

```bash
python src/benchmark.py
```

Benchmark outputs are written to `results/`.

## Project Status

This repository is an agent-systems experimentation platform. It is useful for comparing planning, tool-use, retrieval, and memory strategies under a shared benchmark setup.
