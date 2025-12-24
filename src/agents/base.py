import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AgentResponse:
    """Standardized agent response"""
    answer: str
    trajectory: List[Dict[str, Any]]
    success: bool
    num_steps: int
    error: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(
        self,
        llm_client,
        tool_registry,
        memory_store=None,
        max_steps: int = 10
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.memory = memory_store
        self.max_steps = max_steps
    
    @abstractmethod
    def run(self, query: str) -> AgentResponse:
        """Execute the agent on a query"""
        pass
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call LLM with messages"""
        pass
    
    def _format_tools_for_prompt(self) -> str:
        """Format tool schemas for prompt"""
        tools = self.tools.get_all_schemas()
        formatted = []
        
        for tool in tools:
            formatted.append(
                f"- {tool['name']}: {tool['description']}\n"
                f"  Parameters: {json.dumps(tool['parameters'], indent=2)}"
            )
        
        return "\n".join(formatted)