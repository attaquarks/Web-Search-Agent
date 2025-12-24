"""
Simple API server for the AI Agent frontend
Provides endpoints for chat and metrics
"""
import sys
import os

# Change to src directory to match existing import structure
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your existing modules (now that we're in the right path)
from memory import MemoryStore
from tools import ToolRegistry
from agents.base import BaseAgent
from agents.react_agent import ReActAgent
from agents.plan_execute import PlanExecuteAgent
from agents.plan_execute_memory import PlanExecuteMemoryAgent
from agents.simple_rag import SimpleRAGAgent
from agents.one_shot import OneShotAgent

app = FastAPI(title="AI Agent API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM Client class (copied from main.py to avoid import issues)
class LLMClient:
    """Wrapper for OpenRouter API calls with fallback support"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        # Order of models to use as fallback
        self.models = [
            "xiaomi/mimo-v2-flash:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "arcee-ai/trinity-mini:free",
            "tngtech/tng-r1t-chimera:free"
        ]
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def chat(self, messages, temperature=0.7, max_tokens=2000):
        """Send chat request to LLM via OpenRouter with fallback logic"""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/web-search-agent",
            "X-Title": "Web Search Agent"
        }

        last_error = "Unknown error"
        
        for model in self.models:
            print(f"  🤖 Trying model: {model}...")
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            try:
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=20)
                result = response.json()
                
                if "error" in result:
                    error_msg = result.get("error", {}).get("message", "Unknown API error")
                    print(f"  ⚠️  Model {model} API Error: {error_msg}")
                    last_error = error_msg
                    continue

                response.raise_for_status()
                
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    if content and content.strip() != "":
                        return content
                    else:
                        last_error = "Empty response from LLM"
                        print(f"  ⚠️  Model {model} returned empty response")
                else:
                    last_error = "No choices in response"
                    print(f"  ⚠️  Model {model} failed: {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                print(f"  ❌ Model {model} error: {last_error}")
            
            print(f"  🔄 Attempting fallback to next model...")

        print(f"❌ All models failed. Last error: {last_error}")
        return f"Error: All models failed. Latest error: {last_error}"


# Initialize components
llm_client = LLMClient()
memory_store = MemoryStore()
tool_registry = ToolRegistry(memory_store=memory_store)

# Monkey-patch the _call_llm method for agents
def make_call_llm(client):
    def call_llm(self, messages):
        return client.chat(messages)
    return call_llm

BaseAgent._call_llm = make_call_llm(llm_client)

# Initialize agents
agents = {
    "OneShot": OneShotAgent(llm_client.chat, tool_registry, memory_store),
    "SimpleRAG": SimpleRAGAgent(llm_client.chat, tool_registry, memory_store),
    "ReAct": ReActAgent(llm_client.chat, tool_registry, memory_store),
    "Plan-Execute": PlanExecuteAgent(llm_client.chat, tool_registry, memory_store),
    "Plan-Execute-Memory": PlanExecuteMemoryAgent(llm_client.chat, tool_registry, memory_store),
}


class ChatRequest(BaseModel):
    agent_name: str
    query: str


class ChatResponse(BaseModel):
    agent_name: str
    answer: str
    success: bool
    latency: float
    trajectory: List[Dict]
    memory_hits: List[Dict]


@app.get("/")
async def root():
    return {
        "message": "AI Agent API Server",
        "version": "1.0.0",
        "endpoints": [
            "/api/chat",
            "/api/metrics",
            "/api/agents"
        ]
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint that processes queries through specified agent
    """
    try:
        agent = agents.get(request.agent_name)
        if not agent:
            raise HTTPException(status_code=400, detail=f"Unknown agent: {request.agent_name}")
        
        # Run the agent
        import time
        start_time = time.time()
        
        response = agent.run(request.query)
        
        latency = time.time() - start_time
        
        # Extract trajectory and memory hits from response
        trajectory = []
        memory_hits = []
        
        # Build trajectory from agent response
        if hasattr(response, 'trajectory') and response.trajectory:
            trajectory = response.trajectory
        
        # Extract memory hits if available
        if hasattr(response, 'memory_contexts') and response.memory_contexts:
            memory_hits = response.memory_contexts
        
        return ChatResponse(
            agent_name=request.agent_name,
            answer=response.answer if hasattr(response, 'answer') else str(response),
            success=response.success if hasattr(response, 'success') else True,
            latency=latency,
            trajectory=trajectory,
            memory_hits=memory_hits
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics")
async def get_metrics():
    """
    Get benchmark metrics from comparison_results.json
    """
    try:
        # Use absolute path from current working directory
        results_path = os.path.join("results", "comparison_results.json")
        
        if not os.path.exists(results_path):
            raise HTTPException(status_code=404, detail=f"Metrics file not found at {results_path}")
        
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        # Convert object format to array format
        agents_array = []
        for agent_name, metrics in data.items():
            agents_array.append(metrics)
        
        return {"agents": agents_array}
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/agents")
async def get_agents():
    """
    Get list of available agents
    """
    return {
        "agents": list(agents.keys())
    }


if __name__ == "__main__":
    print("🚀 Starting AI Agent API Server...")
    print("📍 Server running on http://localhost:8000")
    print("📚 API docs available at http://localhost:8000/docs")
    print("\n✨ Available agents:", list(agents.keys()))
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
