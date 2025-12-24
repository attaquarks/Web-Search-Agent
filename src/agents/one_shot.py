from typing import List, Dict, Any
from .base import BaseAgent, AgentResponse


class OneShotAgent(BaseAgent):
    
    def run(self, query: str) -> AgentResponse:
        
        prompt = f"""Answer the following question to the best of your ability based on your internal knowledge.

Question: {query}

Provide a clear and comprehensive answer:"""
        
        messages = [{"role": "user", "content": prompt}]
        answer = self._call_llm(messages)
        
        trajectory = [{
            "step_index": 1,
            "thought": "Generating answer directly from internal knowledge",
            "action": "direct_generation",
            "action_input": {"query": query},
            "observation": answer
        }]
        
        # SIMPLE MEMORY LOGIC: Save final answer summary to memory
        if self.memory:
            save_tool = self.tools.get_tool("save_memory")
            if save_tool:
                save_tool.execute(
                    topic=f"Research on: {query[:100]}",
                    summary=answer[:500],
                    source_url=""
                )
        
        return AgentResponse(
            answer=answer,
            trajectory=trajectory,
            success=True,
            num_steps=1
        )