import re
import json
from typing import List, Dict, Any, Tuple
from .base import BaseAgent, AgentResponse


class PlanExecuteAgent(BaseAgent):
    """
    Plan-and-Execute Agent (Reason-Plan-ReAct style)
    Planner creates high-level plan, Executor executes each step
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory = []
    
    def run(self, query: str) -> AgentResponse:
        """Execute plan-and-execute loop"""
        self.trajectory = []
        
        # Phase 1: Planning
        plan = self._create_plan(query)
        self.trajectory.append({"phase": "planning", "plan": plan})
        
        if not plan:
            return AgentResponse(
                answer="Failed to create plan",
                trajectory=self.trajectory,
                success=False,
                num_steps=0,
                error="PLANNING_FAILED"
            )
        
        # Phase 2: Execute each step
        context = []
        for step_num, step in enumerate(plan, 1):
            if step_num > self.max_steps:
                break
            
            result = self._execute_step(step, context, query, step_num)
            
            step_info = {
                "step": step_num,
                "plan_step": step,
                "result": result
            }
            self.trajectory.append(step_info)
            
            # Add to context
            context.append(f"Step {step_num}: {step}\nResult: {result}")
            
            # Check if we have the answer
            if self._is_complete(result, query, context):
                final_answer = self._synthesize_answer(query, context)
                return AgentResponse(
                    answer=final_answer,
                    trajectory=self.trajectory,
                    success=True,
                    num_steps=step_num
                )
        
        # Synthesize final answer from context
        final_answer = self._synthesize_answer(query, context)
        
        # SIMPLE MEMORY LOGIC: Save final answer summary to memory
        if self.memory:
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
            success=True,
            num_steps=len(plan)
        )
    
    def _create_plan(self, query: str) -> List[str]:
        """Create high-level plan using planner LLM"""
        planner_prompt = f"""You are a research planner. Create a step-by-step plan to answer this question:

Question: {query}

Available tools:
{self._format_tools_for_prompt()}

Create a concise plan with 3-5 high-level steps. Each step should be a clear research action.

Format:
1. [First step]
2. [Second step]
...

Plan:"""
        
        messages = [{"role": "user", "content": planner_prompt}]
        response = self._call_llm(messages)
        
        # Parse plan steps
        steps = []
        for line in response.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                step = re.sub(r'^\d+\.\s*', '', line)
                if step:
                    steps.append(step)
        
        return steps
    
    def _execute_step(self, step: str, context: List[str], query: str, step_num: int) -> str:
        """Execute a single plan step using ReAct-style executor"""
        executor_prompt = f"""You are executing a research step.

Original Question: {query}

Current Step: {step}

Previous Context:
{chr(10).join(context[-3:])}  

Available Tools:
{self._format_tools_for_prompt()}

Execute this step using tools. Format:
Thought: [reasoning]
Action: {{"tool": "tool_name", "parameters": {{}}}}

Your response:"""
        
        messages = [{"role": "user", "content": executor_prompt}]
        response = self._call_llm(messages)
        
        # Parse thought
        thought = ""
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Parse and execute action with improved JSON handling
        action_match = re.search(r'Action:\s*(\{[^}]*\})', response, re.DOTALL)
        if not action_match:
            # Try to find JSON anywhere - be more greedy to catch incomplete JSON
            action_match = re.search(r'Action:\s*(\{.+?)(?=\n\n|Action:|Thought:|$)', response, re.DOTALL)
        
        if action_match:
            json_str = action_match.group(1).strip()
            
            # Auto-fix common LLM JSON errors
            # Count opening and closing braces
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            
            # Add missing closing braces
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
                print(f"🔧 Auto-fixed JSON: added {open_braces - close_braces} closing brace(s)")
            
            try:
                # Try to parse the JSON
                action = json.loads(json_str)
                tool_name = action.get("tool")
                params = action.get("parameters", {})
                
                if not tool_name:
                    print(f"⚠️  No tool name in action: {action}")
                    return "Error: No tool specified in action"
                
                result = self.tools.execute_tool(tool_name, **params)
                
                # Log detailed step
                self.trajectory.append({
                    "step_index": step_num,
                    "thought": thought,
                    "action": tool_name,
                    "action_input": params,
                    "observation": result[:500] + ("..." if len(result) > 500 else "")
                })
                
                return result
                
            except json.JSONDecodeError as e:
                # JSON parsing still failed - try manual extraction
                print(f"⚠️  JSON Parse Error: {e}")
                print(f"Attempted to parse: {json_str}")
                
                # Try to extract tool name and all parameters manually
                tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', json_str)
                
                if tool_match:
                    tool_name = tool_match.group(1)
                    params = {}
                    
                    # Extract all key-value pairs from parameters
                    query_match = re.search(r'"query"\s*:\s*"([^"]+)"', json_str)
                    if query_match:
                        params["query"] = query_match.group(1)
                    
                    max_results_match = re.search(r'"max_results"\s*:\s*(\d+)', json_str)
                    if max_results_match:
                        params["max_results"] = int(max_results_match.group(1))
                    
                    k_match = re.search(r'"k"\s*:\s*(\d+)', json_str)
                    if k_match:
                        params["k"] = int(k_match.group(1))
                    
                    print(f"🔧 Recovered: tool={tool_name}, params={params}")
                    try:
                        result = self.tools.execute_tool(tool_name, **params)
                        self.trajectory.append({
                            "step_index": step_num,
                            "thought": thought,
                            "action": tool_name,
                            "action_input": params,
                            "observation": result[:500] + ("..." if len(result) > 500 else "")
                        })
                        return result
                    except Exception as e2:
                        return f"Error executing recovered action: {str(e2)}"
                
                return f"Error parsing action JSON: {str(e)}"
                
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                return f"Error executing step: {str(e)}"
        
        # No action found
        print(f"⚠️  No action found in response: {response[:200]}")
        return "No action taken"
    
    def _is_complete(self, result: str, query: str, context: List[str]) -> bool:
        """Check if we have enough information to answer"""
        # Simple heuristic: if result contains substantial information
        return len(result) > 100 and not result.startswith("Error")
    
    def _synthesize_answer(self, query: str, context: List[str]) -> str:
        """Synthesize final answer from context"""
        synthesis_prompt = f"""Based on the research conducted, provide a comprehensive answer.

Question: {query}

Research Results:
{chr(10).join(context)}

Provide a clear, well-structured answer:"""
        
        messages = [{"role": "user", "content": synthesis_prompt}]
        return self._call_llm(messages)