from typing import List, Dict, Any
from .base import BaseAgent, AgentResponse


class SimpleRAGAgent(BaseAgent):
    
    def run(self, query: str) -> AgentResponse:
        trajectory = []
        memory_results = ""
        if self.memory:
            search_tool = self.tools.get_tool("search_memory")
            if search_tool:
                memory_results = search_tool.execute(query, k=3)
                trajectory.append({
                    "step_index": 1,
                    "thought": "Checking memory for existing knowledge",
                    "action": "search_memory",
                    "action_input": {"query": query, "k": 3},
                    "observation": memory_results[:500] + "..." if len(memory_results) > 500 else memory_results
                })

        web_results = ""
        search_tool = self.tools.get_tool("web_search")
        if search_tool:
            web_results = search_tool.execute(query, max_results=5)
            trajectory.append({
                "step_index": 2,
                "thought": "Searching web for real-time information",
                "action": "web_search",
                "action_input": {"query": query, "max_results": 5},
                "observation": web_results[:500] + "..." if len(web_results) > 500 else web_results
            })

        prompt = self._build_generation_prompt(query, memory_results, web_results)
        
        messages = [{"role": "user", "content": prompt}]
        answer = self._call_llm(messages)
        
        trajectory.append({
            "step_index": 3,
            "thought": "Synthesizing answer from retrieved context",
            "action": "generate_answer",
            "action_input": {},
            "observation": answer
        })
        
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
            num_steps=3
        )
    
    def _build_generation_prompt(
        self,
        query: str,
        memory_results: str,
        web_results: str
    ) -> str:
        """Build the prompt for answer generation"""
        
        prompt = f"""You are a helpful research assistant. Answer the following question using the provided information.

Question: {query}

"""
        
        if memory_results and "No relevant memories found" not in memory_results:
            prompt += f"""Retrieved from Memory:
{memory_results}

"""
        
        if web_results and "No results found" not in web_results:
            prompt += f"""Web Search Results:
{web_results}

"""
        
        prompt += """Provide a comprehensive, well-structured answer based on the information above.
If the information is insufficient, acknowledge the limitations.

Answer:"""
        
        return prompt