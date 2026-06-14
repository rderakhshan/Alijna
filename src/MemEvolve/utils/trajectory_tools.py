#!/usr/bin/env python
# coding=utf-8

"""
Trajectory Analysis Tools for Memory Evolution
Provides tools for analyzing task execution trajectories
"""

import ast
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterable

from FlashOAgents import Tool
from openai import OpenAI

from ..config import (
    TRAJECTORY_VIEWER_MAX_TASKS,
    TRAJECTORY_SUMMARY_TEMPERATURE,
    MEMORY_DB_MAX_LINES,
    MEMORY_DB_SAMPLE_LINES,
    MEMORY_DB_JSON_LIKE_THRESHOLD,
)

def _safe_len(value: Any) -> int:
    """Compute length of a stringified value without crashing."""
    try:
        return len(str(value))
    except Exception:
        return 0


class TrajectoryFeedbackAggregator:
    """
    Compute multi-dimensional feedback for trajectories and aggregate across tasks.
    
    Metrics are intentionally lightweight and rely only on the stored JSON logs:
    - accuracy: 1/0 based on judgement/score/is_correct/correct
    - steps: total steps; action steps; has plan/summary
    - tool usage: total calls; unique tools; error-like observations (heuristic)
    - memory guidance: count + ratio of steps with memory_guidance
    - text stats: avg obs length for action steps; answer length; question length
    """

    def __init__(self, task_logs_dir: str):
        self.task_logs_dir = Path(task_logs_dir)

    def _load_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        task_file = self.task_logs_dir / f"{task_id}.json"
        if not task_file.exists():
            return None
        with open(task_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _is_task_correct(self, data: dict) -> bool:
        """Shared correctness checker (kept in sync with TrajectoryViewerTool)."""
        if "judgement" in data:
            judgement = data["judgement"]
            if isinstance(judgement, str):
                return judgement.lower() == "correct"
            if isinstance(judgement, bool):
                return judgement
        if "score" in data:
            score = data["score"]
            if isinstance(score, (int, float)):
                return score > 0
            if isinstance(score, str):
                try:
                    return float(score) > 0
                except ValueError:
                    pass
        if "is_correct" in data:
            is_correct = data["is_correct"]
            if isinstance(is_correct, bool):
                return is_correct
            if isinstance(is_correct, str):
                return is_correct.lower() in ["true", "1", "yes"]
        if "correct" in data:
            correct = data["correct"]
            if isinstance(correct, bool):
                return correct
            if isinstance(correct, str):
                return correct.lower() in ["true", "1", "yes"]
        return False

    def _tool_stats(self, trajectory: List[Dict[str, Any]]) -> Tuple[int, int, List[str], int]:
        total_calls = 0
        tool_names = []
        error_calls = 0
        for step in trajectory:
            if step.get("name") != "action":
                continue
            calls = step.get("tool_calls") or []
            total_calls += len(calls)
            for call in calls:
                tool_name = call.get("name") or call.get("tool") or "unknown"
                tool_names.append(tool_name)
            obs = step.get("obs")
            if obs and isinstance(obs, str) and any(err_kw in obs.lower() for err_kw in ["error", "fail", "timeout"]):
                error_calls += 1
        return total_calls, len(set(tool_names)), sorted(set(tool_names)), error_calls

    def _memory_guidance_items(self, step: Dict[str, Any]) -> List[str]:
        """
        Collect memory guidance from the step's memory_guidance field.
        capture_trajectory flattens guidance into step["memory_guidance"], so we only
        look at this field and support string or list payloads.
        """
        val = step.get("memory_guidance")
        if isinstance(val, str) and val.strip():
            return [val.strip()]
        if isinstance(val, list):
            return [str(v).strip() for v in val if str(v).strip()]
        return []

    def _step_success_rate(self, trajectory: List[Dict[str, Any]]) -> float:
        """Fraction of steps without error observations."""
        action_steps = [s for s in trajectory if s.get("name") == "action"]
        if not action_steps:
            return 1.0
        failed = sum(1 for s in action_steps if s.get("obs") and isinstance(s.get("obs"), str) and "error" in s["obs"].lower())
        return 1.0 - (failed / len(action_steps))

    def _tool_invocation_accuracy(self, trajectory: List[Dict[str, Any]]) -> float:
        """Fraction of action steps that invoked at least one tool."""
        action_steps = [s for s in trajectory if s.get("name") == "action"]
        if not action_steps:
            return 1.0
        steps_with_calls = sum(1 for s in action_steps if s.get("tool_calls"))
        return steps_with_calls / len(action_steps)

    def _execution_correctness(self, trajectory: List[Dict[str, Any]]) -> float:
        """Fraction of action steps without any error-like signal in observations."""
        action_steps = [s for s in trajectory if s.get("name") == "action"]
        if not action_steps:
            return 1.0
        error_like = 0
        for s in action_steps:
            obs = str(s.get("obs", ""))
            if any(kw in obs.lower() for kw in ["error", "fail", "exception", "timeout", "traceback"]):
                error_like += 1
        return 1.0 - (error_like / len(action_steps))

    def _detect_prompt_leak(self, text: str) -> float:
        """Returns 1.0 if text contains prompt leakage signals, else 0.0."""
        keywords = [
            "system_prompt", "you are an", "you are a", "your instruction",
            "api_key", "api key", "secret key", "openai_api_key",
            "ignore previous", "ignore all", "forget your",
        ]
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                return 1.0
        return 0.0

    def _validate_ast(self, code: str) -> float:
        """Returns 1.0 if code is syntactically valid Python, else 0.0."""
        try:
            ast.parse(code)
            return 1.0
        except SyntaxError:
            return 0.0

    def _calculate_parameter_f1(self, trajectory: List[Dict[str, Any]], expected_args: Optional[Dict[str, Any]] = None) -> float:
        """Calculate F1 score for generated vs expected tool parameters."""
        if expected_args is None:
            return 1.0
        
        generated_args = {}
        for step in trajectory:
            if step.get("name") == "action":
                for call in (step.get("tool_calls") or []):
                    args = call.get("args") or call.get("arguments") or call.get("input")
                    if isinstance(args, dict):
                        generated_args.update(args)
                    elif isinstance(args, str):
                        try:
                            generated_args.update(json.loads(args))
                        except Exception:
                            pass
                            
        if not expected_args:
            return 1.0 if not generated_args else 0.0
            
        tp = sum(1 for k, v in generated_args.items() if k in expected_args and expected_args[k] == v)
        fp = len(generated_args) - tp
        fn = len(expected_args) - tp
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def _calculate_memory_span(self, data: dict) -> float:
        """Calculate memory span based on length of agent trajectory."""
        trajectory = data.get("agent_trajectory", [])
        return float(len(trajectory))

    def _calculate_token_efficiency(self, data: dict) -> float:
        """Calculate token cost efficiency (tokens per second)."""
        metrics = data.get("metrics", {})
        tokens = metrics.get("total_tokens", 0)
        time_sec = metrics.get("elapsed_time", 1)
        return tokens / max(time_sec, 0.001)

    def _calculate_retrieval_ndcg(self, retrieved: List[str], selected: str) -> float:
        """Calculate tool selection accuracy (NDCG)."""
        import math
        if not retrieved:
            return 0.0
        if retrieved[0] == selected:
            return 1.0
        try:
            idx = retrieved.index(selected)
            return 1.0 / math.log2(idx + 2)
        except ValueError:
            return 0.0

    def _calculate_context_decay(self, data: dict) -> float:
        """Calculate context span decay."""
        history = data.get("metrics", {}).get("context_token_history", [])
        if not history:
            return 0.0
        avg_tokens = sum(history) / len(history)
        return float(history[-1] / avg_tokens) if avg_tokens > 0 else 0.0

    def _calculate_progress_rate(self, data: dict) -> float:
        """Calculate subgoal progress rate."""
        metrics = data.get("metrics", {})
        total = metrics.get("goals_total", 1)
        completed = metrics.get("goals_completed", 0)
        return min(completed / total, 1.0)

    def _calculate_factual_recall(self, actual: list, expected: list) -> float:
        """Calculate factual recall accuracy."""
        if not expected:
            return 1.0
        recalled = sum(1 for item in actual if item in expected)
        return recalled / len(expected)

    def _calculate_policy_adherence(self, answer: str, policy_rules: list) -> float:
        """Calculate policy adherence score."""
        if not policy_rules:
            return 1.0
        passed = sum(1 for rule in policy_rules if rule.lower() in answer.lower())
        return passed / len(policy_rules)

    def _calculate_error_recovery(self, data: dict) -> float:
        """Calculate error recovery rate."""
        metrics = data.get("metrics", {})
        injected = metrics.get("errors_injected", 0)
        recovered = metrics.get("errors_recovered", 0)
        return recovered / injected if injected > 0 else 1.0

    def compute_feedback(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        trajectory: List[Dict[str, Any]] = task_data.get("agent_trajectory", [])
        is_correct = self._is_task_correct(task_data)
        steps_total = len(trajectory)
        action_steps = sum(1 for step in trajectory if step.get("name") == "action")
        plan_steps = sum(1 for step in trajectory if step.get("name") == "plan")
        summary_steps = sum(1 for step in trajectory if step.get("name") == "summary")
        memory_step_items = [self._memory_guidance_items(step) for step in trajectory]
        memory_steps = sum(1 for items in memory_step_items if items)

        tool_total, tool_unique_count, tool_unique_list, tool_error_like = self._tool_stats(trajectory)
        obs_lengths = [
            _safe_len(step.get("obs", ""))
            for step in trajectory
            if step.get("name") == "action"
        ]
        avg_obs_len = sum(obs_lengths) / len(obs_lengths) if obs_lengths else 0.0
        total_memory_items = sum(len(items) for items in memory_step_items)

        # Token usage (optional)
        tokens = {}
        metrics = task_data.get("metrics") or {}
        for key in ["total_tokens", "prompt_tokens", "completion_tokens", "elapsed_time"]:
            if key in task_data:
                tokens[key] = task_data.get(key)
            if key in metrics:
                tokens[key] = metrics.get(key)

        step_success = self._step_success_rate(trajectory)
        tool_invocation_acc = self._tool_invocation_accuracy(trajectory)
        exec_correctness = self._execution_correctness(trajectory)
        prompt_leak = self._detect_prompt_leak(str(task_data.get("agent_result", "")))

        # AST validation
        import re
        agent_result = str(task_data.get("agent_result", ""))
        code_blocks = re.findall(r"```python\s*(.*?)\s*```", agent_result, re.DOTALL)
        if not code_blocks:
            code_blocks = re.findall(r"```\s*(.*?)\s*```", agent_result, re.DOTALL)
        ast_val = 1.0
        if code_blocks:
            ast_val = min(self._validate_ast(block) for block in code_blocks)
        elif "def " in agent_result or "import " in agent_result:
            ast_val = self._validate_ast(agent_result)

        # Parameter F1
        param_f1 = self._calculate_parameter_f1(trajectory, task_data.get("expected_args"))

        # Memory span
        mem_span = self._calculate_memory_span(task_data)

        # Token efficiency
        token_eff = self._calculate_token_efficiency(task_data)

        # Tool retrieval NDCG
        selected_tool = None
        retrieved_tools = []
        for step in trajectory:
            if step.get("name") == "action":
                tool_calls = step.get("tool_calls") or []
                if tool_calls:
                    selected_tool = tool_calls[0].get("name") or tool_calls[0].get("tool")
                    retrieved_tools = step.get("available_tools") or []
                    break
        target_tool = task_data.get("target_tool") or selected_tool
        retrieval_ndcg = self._calculate_retrieval_ndcg(retrieved_tools, target_tool) if target_tool else 1.0

        # Context decay
        ctx_decay = self._calculate_context_decay(task_data)

        # Progress rate
        prog_rate = self._calculate_progress_rate(task_data)

        # Factual recall
        actual_recalls = []
        for step in trajectory:
            actual_recalls.extend(self._memory_guidance_items(step))
        fact_recall = self._calculate_factual_recall(actual_recalls, task_data.get("expected_recalls"))

        # Policy adherence
        policy_adh = self._calculate_policy_adherence(agent_result, task_data.get("policy_rules"))

        # Error recovery
        err_rec = self._calculate_error_recovery(task_data)

        feedback = {
            "accuracy": 1 if is_correct else 0,
            "status": task_data.get("status", "unknown"),
            "score_raw": task_data.get("score"),
            "step_success_rate": step_success,
            "tool_invocation_accuracy": tool_invocation_acc,
            "execution_correctness": exec_correctness,
            "prompt_leak": prompt_leak,
            "ast_validation": ast_val,
            "parameter_f1": param_f1,
            "memory_span": mem_span,
            "token_efficiency": token_eff,
            "retrieval_ndcg": retrieval_ndcg,
            "context_decay": ctx_decay,
            "progress_rate": prog_rate,
            "factual_recall": fact_recall,
            "policy_adherence": policy_adh,
            "error_recovery": err_rec,
            "steps": {
                "total": steps_total,
                "action": action_steps,
                "plan": plan_steps,
                "summary": summary_steps,
                "memory_guidance": {
                    "count": memory_steps,
                    "items": total_memory_items,
                    "ratio": (memory_steps / steps_total) if steps_total else 0.0,
                },
            },
            "tools": {
                "total_calls": tool_total,
                "unique_tools": tool_unique_count,
                "unique_names": tool_unique_list,
                "error_like_calls": tool_error_like,
            },
            "text": {
                "answer_len": _safe_len(task_data.get("agent_result", "")),
                "question_len": _safe_len(task_data.get("question", "")),
                "avg_obs_len": avg_obs_len,
            },
        }
        if tokens:
            feedback["tokens"] = tokens
        return feedback

    def aggregate(self, task_ids: Optional[Iterable[str]] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Aggregate metrics across tasks. If task_ids is None, aggregates all tasks in the dir.
        """
        if task_ids is None:
            task_ids = [p.stem for p in sorted(self.task_logs_dir.glob("*.json"))]
        task_ids = list(task_ids)
        if limit is not None:
            task_ids = task_ids[:limit]

        per_task = []
        for tid in task_ids:
            data = self._load_task(tid)
            if not data:
                continue
            fb = self.compute_feedback(data)
            fb["task_id"] = tid
            per_task.append(fb)

        if not per_task:
            return {"count": 0, "per_task": []}

        def _avg(key_path: List[str]) -> float:
            vals = []
            for item in per_task:
                cur = item
                for k in key_path:
                    cur = cur.get(k, {}) if isinstance(cur, dict) else None
                if isinstance(cur, (int, float)):
                    vals.append(cur)
            return (sum(vals) / len(vals)) if vals else 0.0

        token_keys = ["total_tokens", "prompt_tokens", "completion_tokens", "elapsed_time"]
        has_tokens = any(item.get("tokens") for item in per_task)

        summary = {
            "count": len(per_task),
            "accuracy": _avg(["accuracy"]),
            "step_success_rate": _avg(["step_success_rate"]),
            "tool_invocation_accuracy": _avg(["tool_invocation_accuracy"]),
            "execution_correctness": _avg(["execution_correctness"]),
            "prompt_leak": _avg(["prompt_leak"]),
            "memory_span": _avg(["memory_span"]),
            "ast_validation": _avg(["ast_validation"]),
            "parameter_f1": _avg(["parameter_f1"]),
            "token_efficiency": _avg(["token_efficiency"]),
            "retrieval_ndcg": _avg(["retrieval_ndcg"]),
            "context_decay": _avg(["context_decay"]),
            "progress_rate": _avg(["progress_rate"]),
            "factual_recall": _avg(["factual_recall"]),
            "policy_adherence": _avg(["policy_adherence"]),
            "error_recovery": _avg(["error_recovery"]),
            "steps": {
                "total": _avg(["steps", "total"]),
                "action": _avg(["steps", "action"]),
                "plan": _avg(["steps", "plan"]),
                "summary": _avg(["steps", "summary"]),
                "memory_guidance_ratio": _avg(["steps", "memory_guidance", "ratio"]),
            },
            "tools": {
                "total_calls": _avg(["tools", "total_calls"]),
                "unique_tools": _avg(["tools", "unique_tools"]),
                "error_like_calls": _avg(["tools", "error_like_calls"]),
            },
            "text": {
                "answer_len": _avg(["text", "answer_len"]),
                "question_len": _avg(["text", "question_len"]),
                "avg_obs_len": _avg(["text", "avg_obs_len"]),
            },
        }
        if has_tokens:
            summary["tokens"] = {
                key: _avg(["tokens", key]) for key in token_keys
            }
        return {"summary": summary, "per_task": per_task}


class TrajectoryViewerTool(Tool):
    """View task execution summaries with step-by-step breakdown"""
    
    name = "view_trajectories"
    description = """View detailed information for specific tasks with step-by-step trajectory summaries.
    
    Args:
        task_ids: List of task IDs to view (maximum 3 tasks at once)
    
    Returns for each task:
    - question: Full question text
    - is_correct: Whether agent answered correctly
    - agent_answer: The agent's final answer
    - correct_answer: The expected correct answer
    - trajectory_summary: Step-by-step summary (e.g., "Step 0: ...", "Step 1: ...")
    - total_steps: Number of steps in trajectory
    - elapsed_time: Task execution time in seconds (if available)
    - feedback: Multi-dimensional metrics (accuracy, steps, tools, text stats, tokens)
    
    The trajectory_summary breaks down the execution step by step, making it easy
    to identify which specific steps to examine using view_specific_steps.
    
    Summary caching: First call generates summary and saves it. Future calls use cached version.
    Limited to 3 tasks at once to avoid overwhelming context."""
    
    inputs = {
        "task_ids": {
            "type": "array",
            "description": "List of task IDs to view (maximum 3 tasks at once)"
        }
    }
    output_type = "string"
    
    def __init__(self, task_logs_dir: str, max_tasks: int = TRAJECTORY_VIEWER_MAX_TASKS, model_id: str = "gpt-5-mini"):
        super().__init__()
        self.task_logs_dir = Path(task_logs_dir)
        self.max_tasks = max_tasks
        self.model_id = model_id
        self.feedback_aggregator = TrajectoryFeedbackAggregator(task_logs_dir)
        
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
    
    def _generate_trajectory_summary(self, trajectory: List[Dict], question: str, result: str, is_correct: bool) -> str:
        """Generate step-by-step summary of trajectory using LLM"""
        summary_prompt = f"""Summarize this agent task execution trajectory step by step.

Question: {question}
Final Result: {result}
Correctness: {'Correct' if is_correct else 'Wrong'}

Trajectory ({len(trajectory)} steps):
{json.dumps(trajectory, indent=2)[:5000]}

IMPORTANT: Provide a step-by-step summary in this format:

Step 0: [Brief description of what happened in step 0, including memory guidance if any]
Step 1: [Brief description of what happened in step 1, including tool calls and key observations]
Step 2: [Brief description of what happened in step 2]
...

After the step-by-step summary, add a brief conclusion about:
- Overall approach taken
- Whether memory guidance was effective
- Why the task succeeded or failed

Use actual step indices (0, 1, 2, ...) that match the trajectory array indices.
Keep each step description to 1-2 sentences.
Total length should be under 400 words."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=800,
                temperature=TRAJECTORY_SUMMARY_TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Failed to generate summary: {str(e)}"
    
    def forward(self, task_ids: List[str]) -> str:
        """View trajectories for specified tasks with summary caching"""
        if len(task_ids) > self.max_tasks:
            return json.dumps({
                "error": f"Too many tasks requested. Maximum {self.max_tasks} at once, you requested {len(task_ids)}.",
                "suggestion": "Call this tool multiple times with smaller batches."
            }, indent=2)
        
        results = {}
        
        for task_id in task_ids:
            task_file = self.task_logs_dir / f"{task_id}.json"
            
            if not task_file.exists():
                results[task_id] = {"error": "Task file not found"}
                continue
            
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
                
                trajectory = task_data.get("agent_trajectory", [])
                
                if "trajectory_summary" not in task_data:
                    print(f"  Generating summary for task {task_id}...")
                    summary = self._generate_trajectory_summary(
                        trajectory=trajectory,
                        question=task_data.get("question", ""),
                        result=task_data.get("agent_result", ""),
                        is_correct=self._is_task_correct(task_data)
                    )
                    
                    task_data["trajectory_summary"] = summary
                    
                    with open(task_file, 'w', encoding='utf-8') as f:
                        json.dump(task_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"  Summary cached for task {task_id}")
                else:
                    summary = task_data["trajectory_summary"]
                    print(f"  Using cached summary for task {task_id}")
                
                # Build result with basic info
                result_dict = {
                    "question": task_data.get("question", ""),
                    "is_correct": self._is_task_correct(task_data),
                    "agent_answer": task_data.get("agent_result", ""),
                    "correct_answer": task_data.get("golden_answer", ""),
                    "trajectory_summary": summary,
                    "total_steps": len(trajectory),
                    "feedback": self.feedback_aggregator.compute_feedback(task_data)
                }
                
                # Add execution time if available
                metrics = task_data.get("metrics", {})
                if metrics:
                    elapsed_time = metrics.get("elapsed_time")
                    if elapsed_time is not None:
                        result_dict["elapsed_time"] = elapsed_time
                
                results[task_id] = result_dict
            
            except Exception as e:
                results[task_id] = {"error": f"Failed to load: {str(e)}"}
        
        return json.dumps(results, indent=2)
    
    def _is_task_correct(self, data: dict) -> bool:
        """
        Determine if a task is correct using multiple possible field names and formats.
        
        Supports:
        - judgement: "correct" / "incorrect"
        - score: 1 / 0
        - is_correct: True / False
        - correct: True / False
        """
        # Check 'judgement' field
        if "judgement" in data:
            judgement = data["judgement"]
            if isinstance(judgement, str):
                return judgement.lower() == "correct"
            elif isinstance(judgement, bool):
                return judgement
        
        # Check 'score' field (1 = correct, 0 = incorrect)
        if "score" in data:
            score = data["score"]
            if isinstance(score, (int, float)):
                return score > 0
            elif isinstance(score, str):
                try:
                    return float(score) > 0
                except ValueError:
                    pass
        
        # Check 'is_correct' field
        if "is_correct" in data:
            is_correct = data["is_correct"]
            if isinstance(is_correct, bool):
                return is_correct
            elif isinstance(is_correct, str):
                return is_correct.lower() in ["true", "1", "yes"]
        
        # Check 'correct' field
        if "correct" in data:
            correct = data["correct"]
            if isinstance(correct, bool):
                return correct
            elif isinstance(correct, str):
                return correct.lower() in ["true", "1", "yes"]
        
        # Default: False
        return False


class StepViewerTool(Tool):
    """View detailed information for specific trajectory steps"""
    
    name = "view_specific_steps"
    description = """View the complete details of specific steps in a task's execution trajectory.
    
    Args:
        task_id: Task ID to examine
        step_indices: List of step indices to view (e.g., [0, 2, 5])
    
    Returns for each requested step:
    - name: Step type (plan, action, summary)
    - value: Main content (e.g., plan text, final answer)
    - think: Agent's thinking process
    - cot_think: Chain-of-thought reasoning (if present)
    - memory_guidance: Memory guidance provided at this step (if present)
    - tool_calls: List of tool calls made (for action steps)
    - obs: Observations/results from tools (for action steps)
    
    Use this after reviewing the trajectory_summary from view_trajectories to examine
    specific steps in full detail. Step indices are 0-based and match the summary."""
    
    inputs = {
        "task_id": {
            "type": "string",
            "description": "Task ID to examine"
        },
        "step_indices": {
            "type": "array",
            "description": "List of step indices to view (e.g., [0, 2, 5])"
        }
    }
    output_type = "string"
    
    def __init__(self, task_logs_dir: str):
        super().__init__()
        self.task_logs_dir = Path(task_logs_dir)
    
    def forward(self, task_id: str, step_indices: List[int]) -> str:
        """View specific steps from a trajectory"""
        task_file = self.task_logs_dir / f"{task_id}.json"
        
        if not task_file.exists():
            return json.dumps({
                "error": f"Task file for {task_id} not found"
            }, indent=2)
        
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            
            trajectory = task_data.get("agent_trajectory", [])
            
            steps = {}
            for idx in step_indices:
                if 0 <= idx < len(trajectory):
                    steps[f"step_{idx}"] = trajectory[idx]
                else:
                    steps[f"step_{idx}"] = {
                        "error": f"Step index {idx} out of range (trajectory has {len(trajectory)} steps)"
                    }
            
            return json.dumps({
                "task_id": task_id,
                "total_steps": len(trajectory),
                "requested_steps": steps
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to load task: {str(e)}"
            }, indent=2)


class MemoryDatabaseViewerTool(Tool):
    """View stored memories in the base provider's memory database"""
    
    name = "view_memory_database"
    description = """View the contents of the memory database used by the base provider.
    
    This tool allows you to inspect what memories have been stored by the current memory system.
    You can view a specific range of lines from the database file.
    
    Supports all text-based memory file formats including:
    - JSON (.json): Single JSON object or array
    - JSONL (.jsonl): Line-delimited JSON entries
    - Python (.py): Python source code with memory data
    - Plain text (.txt): Raw text memories
    - Any other text-based format
    
    Args:
        db_path: Path to the memory database file (e.g., "storage/agent_kb/kb_database.json", 
                 "memories.jsonl", "memory_store.py", etc.)
        start_line: Starting line number (1-based, inclusive)
        end_line: Ending line number (1-based, inclusive)
    
    Returns:
    - total_lines: Total number of lines in the database file
    - requested_range: The line range you requested
    - content: The actual content of the requested lines
    - file_type: Detected file type
    - structure_info: Information about the file structure
    
    Constraints:
    - Maximum 200 lines per request
    - Line numbers are 1-based (first line is line 1)
    - Supports all text-based file formats
    
    Usage tips:
    1. First call with a small range (e.g., 1-50) to understand the structure
    2. Use total_lines to determine what ranges are available
    3. Focus on areas with dense memory entries to understand storage patterns"""
    
    inputs = {
        "db_path": {
            "type": "string",
            "description": "Path to the memory database file (supports .json, .jsonl, .py, .txt, and other text formats)"
        },
        "start_line": {
            "type": "integer",
            "description": "Starting line number (1-based, inclusive)"
        },
        "end_line": {
            "type": "integer",
            "description": "Ending line number (1-based, inclusive)"
        }
    }
    output_type = "string"
    
    MAX_LINES_PER_REQUEST = MEMORY_DB_MAX_LINES
    
    def __init__(self):
        super().__init__()
    
    def _detect_file_type_and_structure(self, db_file: Path, lines: list) -> tuple:
        """Detect file type and analyze structure
        
        Returns:
            tuple: (file_type, structure_info)
        """
        file_ext = db_file.suffix.lower()
        file_type = file_ext[1:] if file_ext else "unknown"
        structure_info = ""
        
        try:
            # Try to parse as JSON (single object/array)
            if file_ext == ".json":
                with open(db_file, 'r', encoding='utf-8') as f:
                    db_data = json.load(f)
                
                if isinstance(db_data, dict):
                    structure_info = f"JSON object with keys: {list(db_data.keys())}"
                    if "memories" in db_data:
                        structure_info += f"\n- 'memories' field contains {len(db_data.get('memories', []))} entries"
                    # Count total items in nested structures
                    total_items = sum(len(v) if isinstance(v, list) else 1 for v in db_data.values())
                    structure_info += f"\n- Total items in structure: {total_items}"
                elif isinstance(db_data, list):
                    structure_info = f"JSON array with {len(db_data)} entries"
                return file_type, structure_info
            
            # Try to parse as JSONL (line-delimited JSON)
            elif file_ext == ".jsonl":
                valid_json_lines = 0
                sample_keys = set()
                for i, line in enumerate(lines[:MEMORY_DB_SAMPLE_LINES]):  # Check first N lines
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            valid_json_lines += 1
                            if isinstance(obj, dict):
                                sample_keys.update(obj.keys())
                        except:
                            pass
                
                structure_info = f"JSONL format with {valid_json_lines} valid JSON entries (sampled from first {MEMORY_DB_SAMPLE_LINES} lines)"
                if sample_keys:
                    structure_info += f"\n- Common keys found: {list(sample_keys)[:10]}"
                structure_info += f"\n- Total lines in file: {len(lines)}"
                return file_type, structure_info
            
            # Python file
            elif file_ext == ".py":
                # Count basic Python constructs
                imports = sum(1 for line in lines if line.strip().startswith(('import ', 'from ')))
                functions = sum(1 for line in lines if line.strip().startswith('def '))
                classes = sum(1 for line in lines if line.strip().startswith('class '))
                
                structure_info = f"Python source file"
                structure_info += f"\n- {imports} import statements"
                structure_info += f"\n- {classes} class definitions"
                structure_info += f"\n- {functions} function definitions"
                structure_info += f"\n- {len(lines)} total lines"
                return file_type, structure_info
            
            # Plain text or other formats
            else:
                non_empty_lines = sum(1 for line in lines if line.strip())
                structure_info = f"Text file ({file_type or 'unknown extension'})"
                structure_info += f"\n- {len(lines)} total lines ({non_empty_lines} non-empty)"
                
                # Try to detect if it contains JSON-like content
                json_like_lines = sum(1 for line in lines[:MEMORY_DB_SAMPLE_LINES] if '{' in line or '[' in line)
                if json_like_lines > MEMORY_DB_JSON_LIKE_THRESHOLD:
                    structure_info += f"\n- Contains JSON-like content ({json_like_lines}/{MEMORY_DB_SAMPLE_LINES} sampled lines)"
                
                return file_type or "text", structure_info
                
        except Exception as e:
            return file_type, f"Unable to analyze structure: {str(e)}"
    
    def forward(self, db_path: str, start_line: int, end_line: int) -> str:
        """View specific line range from memory database"""
        
        # Validate path
        db_file = Path(db_path)
        if not db_file.exists():
            return json.dumps({
                "error": f"Database file not found: {db_path}",
                "suggestion": "Please check the path. Common paths include 'storage/agent_kb/kb_database.json', 'memories.jsonl', 'memory_store.py', etc."
            }, indent=2)
        
        # Validate line numbers
        if start_line < 1:
            return json.dumps({
                "error": "start_line must be >= 1 (line numbers are 1-based)"
            }, indent=2)
        
        if end_line < start_line:
            return json.dumps({
                "error": "end_line must be >= start_line"
            }, indent=2)
        
        # Check request size
        requested_lines = end_line - start_line + 1
        if requested_lines > self.MAX_LINES_PER_REQUEST:
            return json.dumps({
                "error": f"Too many lines requested: {requested_lines}",
                "max_allowed": self.MAX_LINES_PER_REQUEST,
                "suggestion": f"Please request at most {self.MAX_LINES_PER_REQUEST} lines at once"
            }, indent=2)
        
        try:
            # Read the entire file
            with open(db_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            
            # Adjust end_line if it exceeds file length
            actual_end_line = min(end_line, total_lines)
            
            # Extract requested lines (convert to 0-based indexing)
            content_lines = lines[start_line - 1:actual_end_line]
            content = ''.join(content_lines)
            
            # Detect file type and analyze structure
            file_type, structure_info = self._detect_file_type_and_structure(db_file, lines)
            
            return json.dumps({
                "db_path": db_path,
                "file_type": file_type,
                "total_lines": total_lines,
                "requested_range": f"{start_line}-{end_line}",
                "actual_range": f"{start_line}-{actual_end_line}",
                "lines_returned": len(content_lines),
                "structure_info": structure_info,
                "content": content
            }, indent=2, ensure_ascii=False)
            
        except UnicodeDecodeError:
            return json.dumps({
                "error": f"File is not a text file or uses unsupported encoding: {db_path}",
                "suggestion": "This tool only supports text-based files (JSON, JSONL, Python, plain text, etc.)"
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "error": f"Failed to read database: {str(e)}"
            }, indent=2)
