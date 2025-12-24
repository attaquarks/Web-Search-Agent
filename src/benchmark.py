import json
import time
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np
import os
import sys
from dotenv import load_dotenv
import re

# Load environment variables FIRST before importing modules that need them
load_dotenv()

# Add src to path if running from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import MemoryStore
from tools import ToolRegistry
from main import LLMClient
from agents.base import BaseAgent, AgentResponse
from agents.react_agent import ReActAgent
from agents.plan_execute import PlanExecuteAgent
from agents.plan_execute_memory import PlanExecuteMemoryAgent
from agents.simple_rag import SimpleRAGAgent
from agents.one_shot import OneShotAgent


class KeyPointEvaluator:
    """
    Evaluates answers based on key-point coverage instead of exact match.
    Inspired by MemoryAgentBench's fact-level evaluation.
    """
    
    @staticmethod
    def normalize_text(text: str) -> str:
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    @staticmethod
    def key_point_matches(answer: str, key_points: List[str]) -> Tuple[int, List[bool]]:
        answer_norm = KeyPointEvaluator.normalize_text(answer)
        matches = []
        
        for kp in key_points:
            kp_norm = KeyPointEvaluator.normalize_text(kp)
            # Check if key point is substring of answer
            matched = kp_norm in answer_norm
            matches.append(matched)
        
        return sum(matches), matches
    
    @staticmethod
    def calculate_f1(answer: str, key_points: List[str]) -> Dict[str, float]:
        if not key_points:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched_count": 0}
        
        matched_count, _ = KeyPointEvaluator.key_point_matches(answer, key_points)
        total_key_points = len(key_points)
        
        recall = matched_count / total_key_points if total_key_points > 0 else 0.0
        precision = recall
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched_count": matched_count,
            "total_key_points": total_key_points
        }


class EvaluationMetrics:
    """
    Metrics for agent evaluation.
    Following principles from MCPVerse, MemoryAgentBench, BrowseComp, and Graphene.
    """
    
    def __init__(self, success_threshold: float = 0.7):
        """
        Args:
            success_threshold: F1 score threshold for considering a task successful
        """
        self.evaluator = KeyPointEvaluator()
        self.success_threshold = success_threshold
    
    def key_point_recall(self, predictions: List[str], key_points_list: List[List[str]]) -> float:
        """Average recall across all questions"""
        recalls = []
        for pred, kps in zip(predictions, key_points_list):
            scores = self.evaluator.calculate_f1(pred, kps)
            recalls.append(scores["recall"])
        return np.mean(recalls) if recalls else 0.0
    
    def key_point_f1(self, predictions: List[str], key_points_list: List[List[str]]) -> float:
        """Average F1 score across all questions"""
        f1_scores = []
        for pred, kps in zip(predictions, key_points_list):
            scores = self.evaluator.calculate_f1(pred, kps)
            f1_scores.append(scores["f1"])
        return np.mean(f1_scores) if f1_scores else 0.0
    
    def task_success_rate(self, predictions: List[str], key_points_list: List[List[str]]) -> float:
        """
        Fraction of tasks where F1 >= threshold.
        Inspired by MCPVerse and AgentBench's task-level success metrics.
        """
        successful = 0
        for pred, kps in zip(predictions, key_points_list):
            scores = self.evaluator.calculate_f1(pred, kps)
            if scores["f1"] >= self.success_threshold:
                successful += 1
        return successful / len(predictions) if predictions else 0.0
    
    @staticmethod
    def avg_steps(trajectories: List[List[Dict]]) -> float:
        """Average number of steps taken (test-time compute proxy)"""
        return np.mean([len(t) for t in trajectories]) if trajectories else 0.0
    
    @staticmethod
    def completion_rate(responses: List[AgentResponse]) -> float:
        """Percentage of queries that completed without errors"""
        successful = sum(1 for r in responses if r.success)
        return successful / len(responses) if responses else 0.0
    
    @staticmethod
    def avg_latency(latencies: List[float]) -> float:
        """Average time per query (user experience and cost proxy)"""
        return np.mean(latencies) if latencies else 0.0
    
    @staticmethod
    def avg_tool_calls(trajectories: List[List[Dict]]) -> float:
        """
        Average number of tool calls per question.
        Tool-use efficiency metric inspired by MCPVerse.
        """
        tool_call_counts = []
        for trajectory in trajectories:
            count = sum(1 for step in trajectory 
                       if step.get('action') and step['action'] != 'finish')
            tool_call_counts.append(count)
        return np.mean(tool_call_counts) if tool_call_counts else 0.0
    
    @staticmethod
    def avg_web_searches(trajectories: List[List[Dict]]) -> float:
        """Average number of web search calls per question"""
        web_search_counts = []
        for trajectory in trajectories:
            count = sum(1 for step in trajectory if step.get('action') == 'web_search')
            web_search_counts.append(count)
        return np.mean(web_search_counts) if web_search_counts else 0.0
    
    @staticmethod
    def tool_use_correctness(trajectories: List[List[Dict]], 
                            test_questions: List[Dict]) -> float:
        """
        Fraction of tasks where agent used tools appropriately.
        
        Correctness criteria:
        - Called web_search when external info was needed (requires_external_info=True)
        - Did NOT call unnecessary tools on trivial factual questions
        """
        correct_usage = 0
        
        for trajectory, question_data in zip(trajectories, test_questions):
            requires_external = question_data.get('requires_external_info', False)
            expected_tools = set(question_data.get('expected_tool_usage', []))
            
            # Get tools actually used
            tools_used = set(step.get('action') for step in trajectory 
                           if step.get('action') and step['action'] != 'finish')
            
            if requires_external:
                # Should have used at least one expected tool
                if expected_tools.intersection(tools_used):
                    correct_usage += 1
            else:
                # Should not have used web_search unnecessarily
                if 'web_search' not in tools_used:
                    correct_usage += 1
                # OR if search_memory was appropriate, that's also fine
                elif 'search_memory' in tools_used:
                    correct_usage += 1
        
        return correct_usage / len(trajectories) if trajectories else 0.0
    
    @staticmethod
    def memory_hit_rate(trajectories: List[List[Dict]]) -> float:
        """
        Fraction of questions where search_memory returned relevant results.
        Inspired by MemoryAgentBench's accurate retrieval (AR) metric.
        """
        total_with_memory_search = 0
        hits = 0
        
        for trajectory in trajectories:
            has_memory_search = False
            for step in trajectory:
                if step.get('action') == 'search_memory':
                    has_memory_search = True
                    observation = step.get('observation', '')
                    # Consider it a hit if observation doesn't indicate no results
                    if observation and 'No relevant memories' not in observation and 'not found' not in observation.lower():
                        hits += 1
                        break  # Count only once per question
            if has_memory_search:
                total_with_memory_search += 1
        
        return hits / total_with_memory_search if total_with_memory_search > 0 else 0.0
    
    @staticmethod
    def redundant_search_reduction(baseline_trajectories: List[List[Dict]], 
                                   memory_trajectories: List[List[Dict]]) -> float:
        """
        Compare web search usage between baseline and memory-augmented agent.
        Positive value means memory agent used fewer searches (better).
        """
        baseline_searches = EvaluationMetrics.avg_web_searches(baseline_trajectories)
        memory_searches = EvaluationMetrics.avg_web_searches(memory_trajectories)
        
        if baseline_searches == 0:
            return 0.0
        
        reduction = (baseline_searches - memory_searches) / baseline_searches
        return reduction


class AgentBenchmark:
    
    def __init__(self, test_questions: List[Dict[str, Any]], success_threshold: float = 0.7):
        self.test_questions = test_questions
        self.metrics = EvaluationMetrics(success_threshold=success_threshold)
        self.success_threshold = success_threshold
    
    def evaluate_agent(
        self,
        agent,
        agent_name: str,
        save_results: bool = True,
        verbose: bool = False
    ) -> Tuple[Dict[str, Any], List[List[Dict[str, Any]]]]:
        """Evaluate a single agent on all test questions"""
        print(f"\n{'='*80}")
        print(f"Evaluating: {agent_name}")
        print(f"{'='*80}")
        
        predictions = []
        responses = []
        latencies = []
        detailed_scores = []
        
        for i, qa in enumerate(tqdm(self.test_questions, desc=f"Testing {agent_name}"), 1):
            question = qa['question']
            key_points = qa['key_points']
            
            # Run agent
            start_time = time.time()
            try:
                response = agent.run(question)
                latency = time.time() - start_time
                
                predictions.append(response.answer)
                responses.append(response)
                latencies.append(latency)
                
                # Calculate detailed scores for this question
                scores = self.metrics.evaluator.calculate_f1(response.answer, key_points)
                detailed_scores.append({
                    "question_id": qa.get('id', i),
                    "question": question,
                    "answer": response.answer,
                    "key_points": key_points,
                    "scores": scores,
                    "success": scores["f1"] >= self.success_threshold,
                    "latency": latency
                })
                
                if verbose and i % 5 == 0:
                    print(f"\n  Q{i}: F1={scores['f1']:.3f}, "
                          f"Recall={scores['recall']:.3f}, "
                          f"Steps={len(response.trajectory)}")
                
            except Exception as e:
                print(f"\n  Error on question {i}: {str(e)}")
                predictions.append("")
                responses.append(AgentResponse(
                    answer=f"Error: {str(e)}",
                    trajectory=[],
                    success=False,
                    num_steps=0
                ))
                latencies.append(0.0)
                detailed_scores.append({
                    "question_id": qa.get('id', i),
                    "question": question,
                    "error": str(e),
                    "scores": {"precision": 0, "recall": 0, "f1": 0},
                    "success": False
                })
        
        # Extract key_points for all questions
        key_points_list = [qa['key_points'] for qa in self.test_questions]
        trajectories = [r.trajectory for r in responses]
        
        # Calculate all metrics
        results = {
            "agent_name": agent_name,
            "total_questions": len(self.test_questions),
            
            # Core quality metrics (outcome-based)
            "key_point_f1": self.metrics.key_point_f1(predictions, key_points_list),
            "key_point_recall": self.metrics.key_point_recall(predictions, key_points_list),
            "task_success_rate": self.metrics.task_success_rate(predictions, key_points_list),
            
            # Efficiency metrics
            "avg_steps": self.metrics.avg_steps(trajectories),
            "avg_tool_calls": self.metrics.avg_tool_calls(trajectories),
            "avg_web_searches": self.metrics.avg_web_searches(trajectories),
            
            # Tool correctness
            "tool_use_correctness": self.metrics.tool_use_correctness(trajectories, self.test_questions),
            
            # Memory-specific metrics
            "memory_hit_rate": self.metrics.memory_hit_rate(trajectories),
            
            # Operational metrics
            "completion_rate": self.metrics.completion_rate(responses),
            "avg_latency": self.metrics.avg_latency(latencies),
            
            # Metadata
            "success_threshold": self.success_threshold
        }
        
        print(f"\n{'='*80}")
        print(f"Results for {agent_name}:")
        print(f"{'='*80}")
        print(f"\n📊 Quality Metrics (Outcome-Based):")
        print(f"  Key-Point F1 Score:    {results['key_point_f1']:.4f}")
        print(f"  Key-Point Recall:      {results['key_point_recall']:.4f}")
        print(f"  Task Success Rate:     {results['task_success_rate']:.4f} (F1 >= {self.success_threshold})")
        
        print(f"\n⚡ Efficiency Metrics:")
        print(f"  Avg Steps/Question:    {results['avg_steps']:.2f}")
        print(f"  Avg Tool Calls:        {results['avg_tool_calls']:.2f}")
        print(f"  Avg Web Searches:      {results['avg_web_searches']:.2f}")
        print(f"  Tool Use Correctness:  {results['tool_use_correctness']:.4f}")
        
        print(f"\n🧠 Memory Metrics:")
        print(f"  Memory Hit Rate:       {results['memory_hit_rate']:.4f}")
        
        print(f"\n⏱️  Operational Metrics:")
        print(f"  Completion Rate:       {results['completion_rate']:.4f}")
        print(f"  Avg Latency:           {results['avg_latency']:.2f}s")
        
        # Save detailed results
        if save_results:
            agent_filename = f"results_{agent_name.lower().replace(' ', '_').replace('-', '_')}.json"
            filename = os.path.join("results", agent_filename)
            with open(filename, 'w') as f:
                detailed = {
                    **results,
                    "detailed_scores": detailed_scores,
                    "trajectories": trajectories
                }
                json.dump(detailed, f, indent=2)
            print(f"\n💾 Detailed results saved to {filename}")
        
        return results, trajectories
    
    def compare_agents(
        self,
        agents: Dict[str, Any],
        save_comparison: bool = True,
        baseline_agent_name: str = None
    ) -> Dict[str, Dict[str, Any]]:
        print("\n" + "="*80)
        print("MULTI-AGENT COMPARISON")
        print("="*80)
        
        all_results = {}
        all_trajectories = {}
        
        for agent_name, agent in agents.items():
            results, trajectories = self.evaluate_agent(agent, agent_name, save_results=True)
            all_results[agent_name] = results
            all_trajectories[agent_name] = trajectories
        
        # Compute search reduction if baseline is specified
        if baseline_agent_name and baseline_agent_name in all_trajectories:
            print(f"\n{'='*80}")
            print(f"Search Reduction Analysis (baseline: {baseline_agent_name})")
            print(f"{'='*80}")
            
            baseline_trajs = all_trajectories[baseline_agent_name]
            for agent_name in all_results:
                if agent_name != baseline_agent_name:
                    reduction = self.metrics.redundant_search_reduction(
                        baseline_trajs,
                        all_trajectories[agent_name]
                    )
                    all_results[agent_name]["search_reduction_vs_baseline"] = reduction
                    print(f"  {agent_name}: {reduction:.2%} reduction in web searches")
        
        # Print comparison table
        print("\n" + "="*140)
        print("COMPREHENSIVE COMPARISON TABLE")
        print("="*140)
        
        metrics = [
            "key_point_f1",
            "task_success_rate",
            "avg_steps",
            "avg_tool_calls",
            "tool_use_correctness",
            "memory_hit_rate",
            "avg_latency"
        ]
        
        metric_labels = {
            "key_point_f1": "F1 Score",
            "task_success_rate": "Success Rate",
            "avg_steps": "Avg Steps",
            "avg_tool_calls": "Tool Calls",
            "tool_use_correctness": "Tool Correct",
            "memory_hit_rate": "Mem Hit Rate",
            "avg_latency": "Latency(s)"
        }
        
        # Header
        print(f"{'Agent':<30}", end="")
        for metric in metrics:
            print(f"{metric_labels[metric]:<18}", end="")
        print()
        print("-" * 140)
        
        # Rows
        for agent_name, results in all_results.items():
            print(f"{agent_name:<30}", end="")
            for metric in metrics:
                value = results.get(metric, 0.0)
                print(f"{value:<18.4f}", end="")
            print()
        
        return all_results

    def generate_comparison_from_results(
        self,
        all_results: Dict[str, Dict[str, Any]],
        all_trajectories: Dict[str, List[List[Dict[str, Any]]]],
        save_comparison: bool = True,
        baseline_agent_name: str = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate comparison from pre-computed results and trajectories.
        """
        print("\n" + "="*80)
        print("GENERIC COMPARISON FROM SAVED RESULTS")
        print("="*80)

        # Compute search reduction if baseline is specified
        if baseline_agent_name and baseline_agent_name in all_trajectories:
            print(f"\n{'='*80}")
            print(f"Search Reduction Analysis (baseline: {baseline_agent_name})")
            print(f"{'='*80}")
            
            baseline_trajs = all_trajectories[baseline_agent_name]
            for agent_name in all_results:
                if agent_name != baseline_agent_name:
                    reduction = self.metrics.redundant_search_reduction(
                        baseline_trajs,
                        all_trajectories[agent_name]
                    )
                    all_results[agent_name]["search_reduction_vs_baseline"] = reduction
                    print(f"  {agent_name}: {reduction:.2%} reduction in web searches")
        
        # Print comparison table
        self.print_comparison_table(all_results)
        
        # Save comparison
        if save_comparison:
            comparison_path = os.path.join("results", "comparison_results.json")
            with open(comparison_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n💾 Comparison saved to {comparison_path}")
        
        # Generate insights
        self._print_insights(all_results)
        
        return all_results

    def print_comparison_table(self, all_results: Dict[str, Dict[str, Any]]):
        """Helper to print the comparison table"""
        print("\n" + "="*140)
        print("COMPREHENSIVE COMPARISON TABLE")
        print("="*140)
        
        metrics = [
            "key_point_f1",
            "task_success_rate",
            "avg_steps",
            "avg_tool_calls",
            "tool_use_correctness",
            "memory_hit_rate",
            "avg_latency"
        ]
        
        metric_labels = {
            "key_point_f1": "F1 Score",
            "task_success_rate": "Success Rate",
            "avg_steps": "Avg Steps",
            "avg_tool_calls": "Tool Calls",
            "tool_use_correctness": "Tool Correct",
            "memory_hit_rate": "Mem Hit Rate",
            "avg_latency": "Latency(s)"
        }
        
        # Header
        print(f"{'Agent':<30}", end="")
        for metric in metrics:
            print(f"{metric_labels[metric]:<18}", end="")
        print()
        print("-" * 140)
        
        # Rows
        for agent_name, results in all_results.items():
            print(f"{agent_name:<30}", end="")
            for metric in metrics:
                value = results.get(metric, 0.0)
                print(f"{value:<18.4f}", end="")
            print()
    
    def _print_insights(self, all_results: Dict[str, Dict[str, Any]]):
        """Print key insights from comparison"""
        print(f"\n{'='*80}")
        print("KEY INSIGHTS")
        print(f"{'='*80}")
        
        # Best F1 score
        best_f1_agent = max(all_results.items(), key=lambda x: x[1]['key_point_f1'])
        print(f"\n🏆 Highest Answer Quality (F1): {best_f1_agent[0]} "
              f"({best_f1_agent[1]['key_point_f1']:.4f})")
        
        # Best success rate
        best_success_agent = max(all_results.items(), key=lambda x: x[1]['task_success_rate'])
        print(f"🎯 Highest Task Success Rate: {best_success_agent[0]} "
              f"({best_success_agent[1]['task_success_rate']:.4f})")
        
        # Most efficient (fewest steps)
        most_efficient = min(all_results.items(), key=lambda x: x[1]['avg_steps'])
        print(f"⚡ Most Efficient (Steps): {most_efficient[0]} "
              f"({most_efficient[1]['avg_steps']:.2f} steps/question)")
        
        # Best tool usage
        best_tool_use = max(all_results.items(), key=lambda x: x[1]['tool_use_correctness'])
        print(f"🔧 Best Tool Use Correctness: {best_tool_use[0]} "
              f"({best_tool_use[1]['tool_use_correctness']:.4f})")
        
        # Best memory utilization
        memory_agents = {k: v for k, v in all_results.items() 
                        if v['memory_hit_rate'] > 0}
        if memory_agents:
            best_memory = max(memory_agents.items(), key=lambda x: x[1]['memory_hit_rate'])
            print(f"🧠 Best Memory Utilization: {best_memory[0]} "
                  f"({best_memory[1]['memory_hit_rate']:.4f})")


def load_test_data(filepath: str) -> List[Dict[str, Any]]:
    """Load test questions from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Validate data structure
        required_fields = ['question', 'key_points']
        for i, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    print(f"Warning: Question {i} missing field '{field}'")
        
        return data
    except FileNotFoundError:
        print(f"Error: Test data file not found at {filepath}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in test data file: {e}")
        return []


if __name__ == "__main__":
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test_questions.json")
    print(f"Loading test data from {data_path}...")
    test_questions = load_test_data(data_path)
    
    if not test_questions:
        print("No test questions found. Exiting.")
        exit(1)
    
    print(f"Loaded {len(test_questions)} test questions")
    
    # 2. Initialize Benchmark (with custom success threshold)
    success_threshold = 0.7  # F1 >= 0.7 for task success
    benchmark = AgentBenchmark(test_questions, success_threshold=success_threshold)
    
    # 3. Initialize Shared Memory
    print("\nInitializing shared memory store...")
    memory_store = MemoryStore()
    memory_path = os.path.join("data", "memory_store.json")
    if os.path.exists(memory_path):
        print(f"Loading existing memory from {memory_path}")
        try:
            memory_store.load_from_file(memory_path)
            print(f"Loaded {len(memory_store.notes)} existing memories")
        except Exception as e:
            print(f"Failed to load memory: {e}")
    
    # 4. Initialize Shared Components
    print("Initializing components...")
    tool_registry = ToolRegistry(memory_store=memory_store)
    llm_client = LLMClient()
    
    # Monkey-patch _call_llm (crucial connection)
    def make_call_llm(client):
        def call_llm(self, messages):
            return client.chat(messages)
        return call_llm
    BaseAgent._call_llm = make_call_llm(llm_client)

    # 5. Initialize Agents
    print("Initializing agents...")
    agents = {
        "OneShot": OneShotAgent(
            llm_client=llm_client,
            tool_registry=tool_registry
        ),
        "SimpleRAG": SimpleRAGAgent(
            llm_client=llm_client,
            tool_registry=tool_registry,
            memory_store=memory_store
        ),
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
        )
    }

    # 6. Parse Command Line Arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run benchmarks for specific agents.")
    parser.add_argument("-react", action="store_true", help="Run ReAct agent")
    parser.add_argument("-plan_execute", action="store_true", help="Run Plan-Execute agent")
    parser.add_argument("-plan_execute_memory", action="store_true", help="Run Plan-Execute-Memory agent")
    parser.add_argument("-simple_rag", action="store_true", help="Run SimpleRAG agent")
    parser.add_argument("-oneshot", action="store_true", help="Run OneShot agent")
    parser.add_argument("-all", action="store_true", help="Run all agents one by one and THEN compare")
    parser.add_argument("--comparison", action="store_true", help="Load existing result files and compare them")
    
    args = parser.parse_args()
    
    # CASE 1: Comparison mode (load from files)
    if args.comparison:
        print("\n" + "="*80)
        print("LOADING EXISTING RESULTS FOR COMPARISON")
        print("="*80)
        
        all_results = {}
        all_trajectories = {}
        
        possible_agents = ["OneShot", "SimpleRAG", "ReAct", "Plan-Execute", "Plan-Execute-Memory"]
        for agent_name in possible_agents:
            agent_filename = f"results_{agent_name.lower().replace(' ', '_').replace('-', '_')}.json"
            filename = os.path.join("results", agent_filename)
            if os.path.exists(filename):
                print(f"  📂 Loading {filename}...")
                with open(filename, 'r') as f:
                    data = json.load(f)
                    # Extract metrics and trajectories
                    metrics_only = {k: v for k, v in data.items() if k not in ["detailed_scores", "trajectories"]}
                    all_results[agent_name] = metrics_only
                    all_trajectories[agent_name] = data.get("trajectories", [])
            else:
                print(f"  ⚠️  No results found for {agent_name} ({filename} missing)")
        
        if not all_results:
            print("❌ No result files found to compare. Run benchmarks first.")
            exit(1)
            
        benchmark.generate_comparison_from_results(
            all_results,
            all_trajectories,
            baseline_agent_name="Plan-Execute" if "Plan-Execute" in all_results else None
        )
        print("\n✅ Comparison mode complete!")
        exit(0)

    # CASE 2: Single agent mode or All agents mode
    # Map flags to agent names
    selected_agents_map = {}
    if args.oneshot: selected_agents_map["OneShot"] = agents["OneShot"]
    if args.simple_rag: selected_agents_map["SimpleRAG"] = agents["SimpleRAG"]
    if args.react: selected_agents_map["ReAct"] = agents["ReAct"]
    if args.plan_execute: selected_agents_map["Plan-Execute"] = agents["Plan-Execute"]
    if args.plan_execute_memory: selected_agents_map["Plan-Execute-Memory"] = agents["Plan-Execute-Memory"]
    
    # If -all or no specific agent selected, we run the comparison logic (which runs all)
    is_single_agent = len(selected_agents_map) > 0 and not args.all
    
    if is_single_agent:
        # Run only the selected agents without full comparison table
        for name, agent in selected_agents_map.items():
            benchmark.evaluate_agent(agent, name, save_results=True)
        print("\n✅ Single agent benchmark complete!")
    else:
        # Run all (either -all was passed or nothing was passed)
        print(f"\n{'='*80}")
        print(f"Starting FULL benchmark evaluation")
        print(f"Agents Total: {len(agents)}")
        print(f"Questions: {len(test_questions)}")
        print(f"Success Threshold: F1 >= {success_threshold}")
        print(f"{'='*80}")
        
        benchmark.compare_agents(
            agents,
            baseline_agent_name="Plan-Execute"
        )
        print("\n✅ Full benchmark complete!")
    
    # 7. Persist Memory
    print(f"\n{'='*80}")
    print(f"Saving memory to {memory_path}...")
    memory_store.save_to_file(memory_path)
    print(f"Memory saved with {len(memory_store.notes)} total memories")
    print(f"{'='*80}")