import os
import json
import requests
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod


class Tool(ABC):
    """Base class for all tools"""
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> str:
        """Execute the tool and return results"""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return tool schema for LLM"""
        pass


class WebSearchTool(Tool):
    """Web search using Tavily API"""
    
    def __init__(self, api_key: Optional[str] = None):
        from tavily import TavilyClient
        self.client = TavilyClient(api_key=api_key or os.getenv("TAVILY_API_KEY"))
    
    def execute(self, query: str, max_results: int = 5) -> str:
        """Search the web and return results"""
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced"
            )
            
            results = []
            for idx, result in enumerate(response.get('results', []), 1):
                results.append(
                    f"[{idx}] {result['title']}\n"
                    f"URL: {result['url']}\n"
                    f"Snippet: {result['content'][:300]}...\n"
                )
            
            return "\n".join(results) if results else "No results found."
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "web_search",
            "description": "Search the web for information using a search engine",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }


class WebPageFetcherTool(Tool):
    """Fetch and parse web page content"""
    
    def execute(self, url: str, max_length: int = 2000) -> str:
        """Fetch webpage and extract clean text"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)
            
            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length] + "...[truncated]"
            
            return f"Content from {url}:\n{text}"
        except Exception as e:
            return f"Error fetching {url}: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "fetch_page",
            "description": "Fetch and extract text content from a web page URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum content length",
                        "default": 2000
                    }
                },
                "required": ["url"]
            }
        }


class MemorySearchTool(Tool):
    """Search through agent's long-term memory"""
    
    def __init__(self, memory_store):
        self.memory = memory_store
    
    def execute(self, query: str, k: int = 3) -> str:
        """Search memory for relevant notes"""
        print(f"\n🔍 SEARCH_MEMORY CALLED: '{query[:50]}'")
        results = self.memory.retrieve_similar(query, k=k)
        
        if not results:
            print(f"❌ No memories found (total in store: {len(self.memory.notes)})")
            return "No relevant memories found."
        
        print(f"✅ Found {len(results)} memories")
        output = []
        for idx, note in enumerate(results, 1):
            output.append(
                f"[Memory {idx}]\n"
                f"Topic: {note.get('topic', 'N/A')}\n"
                f"Summary: {note.get('summary', 'N/A')}\n"
                f"Source: {note.get('source_url', 'N/A')}\n"
            )
        
        return "\n".join(output)
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "search_memory",
            "description": "Search through stored memories for relevant information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for memory"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }


class MemorySaveTool(Tool):
    """Save information to long-term memory"""
    
    def __init__(self, memory_store):
        self.memory = memory_store
    
    def execute(self, topic: str, summary: str, source_url: str = "") -> str:
        """Save a note to memory"""
        print(f"\n🧠 SAVE_MEMORY: '{topic[:50]}'")
        try:
            note_id, is_new = self.memory.add_note(
                topic=topic,
                summary=summary,
                source_url=source_url
            )
            if is_new:
                print(f"✅ Saved! Total notes: {len(self.memory.notes)}")
                return f"Successfully saved to memory: {topic}"
            else:
                print(f"ℹ️  Same information already in memory. Skipping save.")
                return f"Information about '{topic}' already exists."
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return f"Error saving memory: {str(e)}"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "save_memory",
            "description": "Save important information to long-term memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic or title of the memory"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Summary of the information to remember"
                    },
                    "source_url": {
                        "type": "string",
                        "description": "Source URL (optional)",
                        "default": ""
                    }
                },
                "required": ["topic", "summary"]
            }
        }


class ToolRegistry:
    """Registry for managing all available tools"""
    
    def __init__(self, memory_store=None):
        self.tools = {}
        self.memory_store = memory_store
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize all tools"""
        self.register_tool("web_search", WebSearchTool())
        self.register_tool("fetch_page", WebPageFetcherTool())
        
        if self.memory_store:
            self.register_tool("search_memory", MemorySearchTool(self.memory_store))
            self.register_tool("save_memory", MemorySaveTool(self.memory_store))
    
    def register_tool(self, name: str, tool: Tool):
        """Register a tool"""
        self.tools[name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def execute_tool(self, name: str, **kwargs) -> str:
        """Execute a tool with given parameters"""
        tool = self.get_tool(name)
        if not tool:
            return f"Error: Tool '{name}' not found"
        
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools"""
        return [tool.get_schema() for tool in self.tools.values()]