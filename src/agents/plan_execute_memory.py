import re
import json
from typing import List, Dict, Any, Tuple, Optional
from .base import BaseAgent, AgentResponse


class PlanExecuteMemoryAgent(BaseAgent):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory = []
    
    def run(self, query: str) -> AgentResponse:
        """Execute plan-and-execute loop with memory integration"""
        self.trajectory = []
        
        # ============================================================
        # PHASE 1: Memory-Enhanced Planning
        # ============================================================
        print("\n[Phase 1] Planning with memory context...")
        
        # Search memory for relevant past experiences
        memory_context = self._retrieve_planning_memory(query)
        self.trajectory.append({
            "step_index": 0,
            "phase": "planning",
            "thought": "Checking memory for prior knowledge before planning",
            "action": "search_memory",
            "action_input": {"query": query, "k": 5},
            "observation": memory_context[:500] + ("..." if len(memory_context) > 500 else "")
        })
        
        # Create plan with memory insights
        plan = self._create_plan_with_memory(query, memory_context)
        self.trajectory.append({
            "phase": "planning",
            "action": "plan_creation",
            "plan": plan,
            "num_steps": len(plan)
        })
        
        if not plan:
            return AgentResponse(
                answer="Failed to create plan",
                trajectory=self.trajectory,
                success=False,
                num_steps=0,
                error="PLANNING_FAILED"
            )
        
        print(f"  ✓ Created {len(plan)}-step plan")
        
        # ============================================================
        # PHASE 2: Memory-Enhanced Execution
        # ============================================================
        print("\n[Phase 2] Executing with memory integration...")
        
        context = []
        for step_num, step in enumerate(plan, 1):
            if step_num > self.max_steps:
                break
            
            print(f"\n  Step {step_num}/{len(plan)}: {step[:60]}...")
            
            # Execute step with memory
            result = self._execute_step_with_memory(
                step=step,
                step_num=step_num,
                context=context,
                query=query
            )
            
            step_info = {
                "step": step_num,
                "phase": "execution",
                "plan_step": step,
                "result": result[:300] + "..." if len(result) > 300 else result
            }
            self.trajectory.append(step_info)
            
            # Add to context
            context.append(f"Step {step_num}: {step}\nResult: {result}")
            
            # Check if we can answer now
            if self._is_complete(result, query, context):
                print(f"  ✓ Sufficient information gathered")
                break
        
        # ============================================================
        # PHASE 3: Synthesis with Memory Update
        # ============================================================
        print("\n[Phase 3] Synthesizing answer and updating memory...")
        
        final_answer = self._synthesize_answer_with_memory(query, context)
        
        # SIMPLE MEMORY LOGIC: Save final answer summary to memory
        if self.memory:
            save_tool = self.tools.get_tool("save_memory")
            if save_tool:
                save_tool.execute(
                    topic=f"Research on: {query[:100]}",
                    summary=final_answer[:500],
                    source_url=""
                )
        
        self.trajectory.append({
            "phase": "synthesis",
            "action": "final_answer",
            "result": final_answer
        })
        
        print("  ✓ Research complete\n")
        
        return AgentResponse(
            answer=final_answer,
            trajectory=self.trajectory,
            success=True,
            num_steps=len([t for t in self.trajectory if t.get("phase") == "execution"])
        )
    
    # ========================================================================
    # Memory-Enhanced Planning Methods
    # ========================================================================
    
    def _retrieve_planning_memory(self, query: str) -> str:
        """Retrieve relevant memories to inform planning"""
        if not self.memory:
            return ""
        
        search_tool = self.tools.get_tool("search_memory")
        if not search_tool:
            return ""
        
        # Search for relevant past research
        memory_results = search_tool.execute(query, k=5)
        
        if "No relevant memories found" in memory_results:
            return ""
        
        return memory_results
    
    def _create_plan_with_memory(self, query: str, memory_context: str) -> List[str]:
        """Create high-level plan incorporating memory insights"""
        
        planner_prompt = f"""You are a strategic research planner. Create a step-by-step plan to answer this question.

Question: {query}

"""
        
        if memory_context:
            planner_prompt += f"""Relevant Past Research:
{memory_context}

Consider these past findings when planning. You may be able to skip some research steps if relevant information is already in memory.

"""
        
        planner_prompt += f"""Available Tools:
{self._format_tools_for_prompt()}

Create a concise plan with 3-5 high-level research steps. Each step should be a clear action.

Important:
- Check memory first before searching the web
- Build on existing knowledge when possible
- Focus on filling knowledge gaps

Format your plan as:
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
    
    # ========================================================================
    # Memory-Enhanced Execution Methods
    # ========================================================================
    
    def _execute_step_with_memory(
        self,
        step: str,
        step_num: int,
        context: List[str],
        query: str
    ) -> str:
        """Execute a single step with memory integration"""
        
        # BEFORE: Retrieve step-relevant memory
        step_memory = self._retrieve_step_memory(step)
        
        # Build executor prompt with memory context
        executor_prompt = f"""You are executing a research step.

Original Question: {query}

Current Step ({step_num}): {step}

"""
        
        if step_memory:
            executor_prompt += f"""Relevant Memory for This Step:
{step_memory}

"""
        
        if context:
            executor_prompt += f"""Previous Steps Context:
{chr(10).join(context[-2:])}

"""
        
        executor_prompt += f"""Available Tools:
{self._format_tools_for_prompt()}

Execute this step using tools. Use the ReAct format:

Thought: [Your reasoning about what to do]
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
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            
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
                    
                    # Extract all key-value pairs
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
        print(f"⚠️  No action found in response")
        return "No action taken"
    
    def _retrieve_step_memory(self, step: str) -> str:
        """Retrieve memory relevant to current step"""
        if not self.memory:
            return ""
        
        search_tool = self.tools.get_tool("search_memory")
        if not search_tool:
            return ""
        
        # Search for step-specific memories
        memory_results = search_tool.execute(step, k=3)
        
        if "No relevant memories found" in memory_results:
            return ""
        
        return memory_results
    
    def _save_step_learnings(self, step: str, result: str, query: str):
        """Deprecated: using simplified global save at end of run"""
        pass
    
    # ========================================================================
    # Synthesis Methods
    # ========================================================================
    
    def _is_complete(self, result: str, query: str, context: List[str]) -> bool:
        """Check if we have sufficient information"""
        # Simple heuristic: substantial result without errors
        return (
            len(result) > 150 and 
            not result.startswith("Error") and
            len(context) >= 2
        )
    
    def _synthesize_answer_with_memory(self, query: str, context: List[str]) -> str:
        """Synthesize final answer incorporating all context"""
        
        # Also check memory one final time for relevant info
        final_memory_check = ""
        if self.memory:
            search_tool = self.tools.get_tool("search_memory")
            if search_tool:
                final_memory_check = search_tool.execute(query, k=3)
        
        synthesis_prompt = f"""Synthesize a comprehensive answer based on all research conducted.

Question: {query}

Research Context:
{chr(10).join(context)}

"""
        
        if final_memory_check and "No relevant memories" not in final_memory_check:
            synthesis_prompt += f"""Additional Relevant Memory:
{final_memory_check}

"""
        
        synthesis_prompt += """Provide a clear, well-structured, and comprehensive answer.
Include specific details and citations where appropriate.

Answer:"""
        
        messages = [{"role": "user", "content": synthesis_prompt}]
        return self._call_llm(messages)
    
    def _save_research_summary(self, query: str, answer: str, context: List[str]):
        """Deprecated: using simplified global save at end of run"""
        pass