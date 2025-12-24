import re
import json
from typing import List, Dict, Any
from .base import BaseAgent, AgentResponse


class ReActAgent(BaseAgent):
    """
    ReAct Agent: Thought -> Action -> Observation loop
    Based on the ReAct paper (Yao et al. 2023)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory = []
    
    def run(self, query: str) -> AgentResponse:
        """Execute ReAct loop: Search Memory -> Web Search -> Finish -> Save Answer"""
        self.trajectory = []
        visited_tools = set() # To prevent the agent from looping on the same query
        
        system_prompt = self._get_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}"}
        ]
        
        final_answer = None
        for step in range(self.max_steps):
            response = self._call_llm(messages)
            thought, action, finish = self._parse_response(response)
            
            step_info = {
                "step_index": step + 1,
                "thought": thought,
                "action": action.get("tool") if action else "finish",
                "action_input": action.get("parameters") if action else {},
                "observation": None
            }
            
            if finish:
                final_answer = finish
                self.trajectory.append(step_info)
                break
            
            if action:
                tool_name = action.get("tool")
                params = action.get("parameters", {})
                
                # LOOP PREVENTION: If the agent calls the exact same thing twice, force web search or finish
                call_id = f"{tool_name}:{json.dumps(params, sort_keys=True)}"
                if call_id in visited_tools:
                    observation = f"Error: You have already used {tool_name} with these parameters. If memory search returned results, incorporate them. If it failed, use web_search. DO NOT repeat yourself."
                else:
                    visited_tools.add(call_id)
                    observation = self._execute_action(action)
                
                step_info["observation"] = observation
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                # Nudge the agent if it fails to produce a valid action
                messages.append({"role": "user", "content": "I couldn't parse your Action. Please use the format Action: {\"tool\": \"...\", \"parameters\": {...}}"})
            
            self.trajectory.append(step_info)
        
        if not final_answer:
            final_answer = "Max steps reached without finding answer. Please check the research trajectory for details."

        # SIMPLE MEMORY LOGIC: Save final answer summary to memory
        if self.memory and final_answer != "Max steps reached without finding answer. Please check the research trajectory for details.":
            save_tool = self.tools.get_tool("save_memory")
            if save_tool:
                save_tool.execute(
                    topic=f"Research on: {query[:100]}",
                    summary=final_answer[:500],
                    source_url=""
                )
        
        return AgentResponse(
            answer=final_answer,
            trajectory=self.trajectory,
            success=final_answer != "Max steps reached without finding answer. Please check the research trajectory for details.",
            num_steps=len(self.trajectory)
        )
    
    def _get_system_prompt(self) -> str:
        """Get ReAct system prompt"""
        tools_description = self._format_tools_for_prompt()
        
        return f"""You are a helpful and efficient research assistant. Your task is to answer the user's question accurately.

Available Tools:
{tools_description}

Format:
Thought: What do I need to do?
Action: {{"tool": "tool_name", "parameters": {{...}}}}
Observation: [Result from tool]
... (repeat until finished)
Thought: I have enough information.
Finish: [The complete, detailed answer to the user's question]

STRICT RULES:
1. SEARCH MEMORY FIRST: Always check `search_memory` once at the very beginning.
2. EVALUATE: If memory contains the answer, synthesis it and use 'Finish'.
3. WEB SEARCH: If memory is empty or insufficient, use `web_search`.
4. NO LOOPING: Do not call the same tool with the same parameters twice.
5. NO HALLUCINATION: Only use information provided in observations or your internal knowledge if it's general fact. For specific research, rely on tools.
6. STAY ON TOPIC: Focus purely on the user's question.

Begin!"""
    
    def _parse_response(self, response: str):
        """Parse LLM response into thought, action, finish"""
        thought = None
        action = None
        finish = None
        
        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Finish:|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Extract action with improved JSON handling
        action_match = re.search(r'Action:\s*(\{[^}]*\})', response, re.DOTALL)
        if not action_match:
            # Try to find JSON anywhere - be more greedy
            action_match = re.search(r'Action:\s*(\{.+?)(?=\n\n|Finish:|Thought:|$)', response, re.DOTALL)
        
        if action_match:
            json_str = action_match.group(1).strip()
            
            # Auto-fix common LLM JSON errors
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
                print(f"🔧 Auto-fixed JSON: added {open_braces - close_braces} closing brace(s)")
            
            try:
                action = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON Parse Error in ReAct: {e}")
                print(f"Attempted to parse: {json_str}")
                
                # Try to extract tool name and parameters manually
                tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', json_str)
                
                if tool_match:
                    action = {"tool": tool_match.group(1), "parameters": {}}
                    
                    # Extract all parameters
                    query_match = re.search(r'"query"\s*:\s*"([^"]+)"', json_str)
                    if query_match:
                        action["parameters"]["query"] = query_match.group(1)
                    
                    max_results_match = re.search(r'"max_results"\s*:\s*(\d+)', json_str)
                    if max_results_match:
                        action["parameters"]["max_results"] = int(max_results_match.group(1))
                    
                    k_match = re.search(r'"k"\s*:\s*(\d+)', json_str)
                    if k_match:
                        action["parameters"]["k"] = int(k_match.group(1))
                    
                    print(f"🔧 Recovered action: {action}")
                else:
                    action = None
        
        # Extract finish
        finish_match = re.search(r'Finish:\s*(.+?)$', response, re.DOTALL)
        if finish_match:
            finish = finish_match.group(1).strip()
        
        return thought, action, finish
    
    def _execute_action(self, action: Dict[str, Any]) -> str:
        """Execute a tool action"""
        tool_name = action.get("tool")
        parameters = action.get("parameters", {})
        
        return self.tools.execute_tool(tool_name, **parameters)