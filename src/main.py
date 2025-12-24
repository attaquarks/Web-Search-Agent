import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
import json
import requests

from memory import MemoryStore
from tools import ToolRegistry
from agents.base import BaseAgent
from agents.react_agent import ReActAgent
from agents.plan_execute import PlanExecuteAgent
from agents.plan_execute_memory import PlanExecuteMemoryAgent
from agents.simple_rag import SimpleRAGAgent
from agents.one_shot import OneShotAgent


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
                # Use a slightly shorter timeout (20s) to trigger fallback faster
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=20)
                
                # Check for explicit error field in JSON even if status is 200
                result = response.json()
                if "error" in result:
                    error_msg = result.get("error", {}).get("message", "Unknown API error")
                    print(f"  ⚠️  Model {model} API Error: {error_msg}")
                    last_error = error_msg
                    continue # Fallback

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
                    
            except requests.exceptions.HTTPError as e:
                try:
                    error_json = e.response.json()
                    last_error = error_json.get("error", {}).get("message", str(e))
                except:
                    last_error = str(e)
                print(f"  ❌ Model {model} HTTP Error: {last_error}")
                
            except requests.exceptions.Timeout:
                last_error = "Timeout"
                print(f"  ⏱️  Model {model} timed out after 20 seconds")
                
            except Exception as e:
                last_error = str(e)
                print(f"  ❌ Model {model} unexpected error: {last_error}")
            
            print(f"  🔄 Attempting fallback to next model...")

        print(f"❌ All models failed. Last error: {last_error}")
        return f"Error: All models failed. Latest error: {last_error}"




def main():
    """Main execution function"""
    load_dotenv()
    
    print("=" * 80)
    print("Web Research Agent with Memory")
    print("=" * 80)
    
    # Initialize components
    print("\n[1/4] Initializing memory store...")
    memory_store = MemoryStore()
    
    # Load existing memory if available
    memory_path = os.path.join("data", "memory_store.json")
    if os.path.exists(memory_path):
        try:
            print(f"  📂 Loading existing memory from {memory_path}...")
            memory_store.load_from_file(memory_path)
            print(f"  ✅ Loaded {len(memory_store.notes)} existing memories")
        except Exception as e:
            print(f"  ⚠️  Failed to load memory: {e}")
            print(f"  Starting with empty memory store")
    else:
        print(f"  ℹ️  No existing memory found, starting fresh")
    
    print("[2/4] Initializing tools...")
    tool_registry = ToolRegistry(memory_store=memory_store)
    
    print("[3/4] Initializing LLM client (OpenRouter Fallback)...")
    llm_client = LLMClient()
    
    # Monkey-patch the _call_llm method for agents
    def make_call_llm(client):
        def call_llm(self, messages):
            return client.chat(messages)
        return call_llm
    
    BaseAgent._call_llm = make_call_llm(llm_client)
    
    print("[4/4] Initializing agents...")
    
    # Create agents
    agents = {
        "ReAct": ReActAgent(
            llm_client=llm_client,
            tool_registry=tool_registry,
            memory_store=memory_store,
            max_steps=10
        ),
        "Plan-Execute": PlanExecuteAgent(
            llm_client=llm_client,
            tool_registry=tool_registry,
            memory_store=memory_store,
            max_steps=10
        ),
        "Plan-Execute-Memory": PlanExecuteMemoryAgent(
            llm_client=llm_client,
            tool_registry=tool_registry,
            memory_store=memory_store,
            max_steps=15
        ),
        "SimpleRAG": SimpleRAGAgent(
            llm_client=llm_client,
            tool_registry=tool_registry,
            memory_store=memory_store
        ),
        "OneShot": OneShotAgent(
            llm_client=llm_client,
            tool_registry=tool_registry,
            memory_store=memory_store  # OneShot doesn't use memory, but base class init accepts it
        )
    }
    
    print("\n✓ Initialization complete!")
    print("\nAvailable agents:")
    for name in agents.keys():
        print(f"  - {name}")
    
    # Interactive loop
    print("\n" + "=" * 80)
    print("Enter your research questions (or 'quit' to exit)")
    print("=" * 80)
    
    while True:
        print("\n")
        query = input("Question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        # Choose agent
        print("\nSelect agent:")
        for i, name in enumerate(agents.keys(), 1):
            print(f"{i}. {name}")
        
        choice = input("\nChoice (1-5): ").strip() or "1"
        
        try:
            agent_name = list(agents.keys())[int(choice) - 1]
            agent = agents[agent_name]
        except (ValueError, IndexError):
            print("Invalid choice, using ReAct")
            agent_name = "ReAct"
            agent = agents[agent_name]
        
        print(f"\n🤖 Running {agent_name} agent...\n")
        
        # Run agent
        response = agent.run(query)
        
        # Display results
        print("\n" + "=" * 80)
        print(f"ANSWER ({agent_name}):")
        print("=" * 80)
        print(response.answer)
        
        print("\n" + "-" * 80)
        print(f"Steps: {response.num_steps} | Success: {response.success}")
        
        if response.error:
            print(f"Error: {response.error}")
        
        # Show trajectory
        show_traj = input("\nShow trajectory? (y/n) [n]: ").strip().lower()
        if show_traj == 'y':
            print("\n" + "=" * 80)
            print("TRAJECTORY:")
            print("=" * 80)
            for step in response.trajectory:
                print(json.dumps(step, indent=2))
                print("-" * 40)
    
    # Save memory before exit
    print("\n💾 Saving memory...")
    memory_path = os.path.join("data", "memory_store.json")
    memory_store.save_to_file(memory_path)
    print(f"✓ Memory saved to {memory_path}")
    print("\nGoodbye!")


if __name__ == "__main__":
    main()