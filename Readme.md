# AI Agent System: Multi-Agent Search & Research Platform

A comprehensive platform designed to demonstrate and evaluate various AI agent architectures. This project implements multiple search-oriented agent patterns, providing both a backend evaluation framework and a modern web-based interaction interface.

## 🚀 Overview

This system allows you to compare different reasoning patterns in AI agents. Each agent is designed to solve complex research tasks by interacting with web search tools and a long-term memory store. The platform provides real-time visualization of the agent's "thought process" and a detailed dashboard for benchmarking performance.

## 🤖 Supported Agents

- **One-Shot**: Direct answer generation for simple queries.
- **Simple RAG**: Retrieval-Augmented Generation using local knowledge.
- **ReAct**: An iterative "Reason + Act" pattern for multi-step tasks.
- **Plan-Execute**: Strategic planning followed by sequential execution.
- **Plan-Execute + Memory**: Advanced planning with long-term memory retrieval and persistence.

## ✨ Key Features

- **Interactive Playground**: Chat with any agent and switch between them in real-time.
- **Reasoning Trajectory**: Live view of an agent's internal thoughts, tool calls, and observations.
- **Memory Inspector**: Visualize how agents retrieve and store contextual information.
- **Evaluation Dashboard**: Comprehensive metrics (F1 Score, Latency, Tool Efficiency) visualized through tables and charts.
- **Benchmarking Suite**: Automatic evaluation of agents against standardized question sets.

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI, LangChain, OpenRouter (LLM API), FAISS/ChromaDB (Vector Stores).
- **Frontend**: Next.js 14, Tailwind CSS 4, Recharts, Lucide React.
- **Storage**: JSON-based long-term memory and result persistence.

## 📋 Quick Start

### 1. Prerequisites
- Python 3.8+
- Node.js 18+
- OpenRouter API Key in a `.env` file (`OPENROUTER_API_KEY=...`)

### 2. Backend & API Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python api_server.py
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

The application will be available at `http://localhost:3000` and the API at `http://localhost:8000`.

## 📁 Project Structure

- `src/`: Core logic for agents, tools, and evaluation.
- `api_server.py`: FastAPI bridge between backend agents and the web UI.
- `frontend/`: Next.js application (Chat and Dashboard).
- `results/`: Cached benchmarking results for all agents.
- `data/`: Evaluation questions and memory storage.

---