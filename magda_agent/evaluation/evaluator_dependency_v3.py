import json
import logging
from typing import Dict, Any, List, Optional
from magda_agent.llm_client import LLMClient
from magda_agent.planning.dependency_graph import DependencyGraph

class EvaluatorDependencyV3:
    """
    Evaluator module to understand and process task dependency graphs when scoring Planner outputs.
    Inspired by Claude Agent Teams trend (June 2026).
    """
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def evaluate_plan_dependencies(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates the dependencies of a given plan.

        Args:
            plan (Dict[str, Any]): The plan object containing 'steps'.

        Returns:
            Dict[str, Any]: Evaluation result containing score, approved status, and feedback.
        """
        steps = plan.get("steps", [])
        if not steps:
            return {
                "score": 0,
                "approved": False,
                "feedback": "Plan has no steps."
            }

        # 1. Structural Validation & Cycle Detection
        try:
            DependencyGraph.topological_sort(steps)
            structural_valid = True
            structural_feedback = "No cycles detected in dependency graph."
        except ValueError as e:
            structural_valid = False
            structural_feedback = f"Dependency cycle detected: {e}"

        # 2. Missing Dependency Detection
        step_ids = {step.get("id") for step in steps if step.get("id")}
        missing_deps = []
        for step in steps:
            for dep in step.get("dependencies", []):
                if dep not in step_ids:
                    missing_deps.append((step.get("id"), dep))

        if missing_deps:
            structural_valid = False
            structural_feedback += f" Missing dependencies: {missing_deps}"

        # 3. Parallelism Ratio
        total_steps = len(steps)
        # Steps with no dependencies can start immediately
        independent_steps = [s for s in steps if not s.get("dependencies")]
        parallelism_ratio = len(independent_steps) / total_steps if total_steps > 0 else 0

        # 4. LLM Logical Soundness Evaluation
        logical_eval = await self._evaluate_logical_soundness(plan)

        approved = structural_valid and logical_eval.get("approved", False)

        combined_feedback = f"{structural_feedback}\nLogical Evaluation: {logical_eval.get('feedback')}"

        return {
            "score": logical_eval.get("score", 0) if approved else min(logical_eval.get("score", 0), 5),
            "approved": approved,
            "feedback": combined_feedback,
            "parallelism_ratio": parallelism_ratio,
            "metadata": {
                "structural_valid": structural_valid,
                "logical_eval": logical_eval,
                "missing_deps": missing_deps
            }
        }

    async def _evaluate_logical_soundness(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Uses LLM to evaluate if the dependency order makes logical sense for the goal."""
        goal = plan.get("goal", "No goal specified")
        steps = plan.get("steps", [])

        prompt = (
            "You are an expert systems architect. Review the following task dependency graph for an AI agent's plan.\n"
            f"Goal: {goal}\n"
            "Steps and Dependencies:\n"
        )
        for step in steps:
            deps = ", ".join(step.get("dependencies", []))
            prompt += f"- Step ID: {step.get('id')}\n  Description: {step.get('description')}\n  Dependencies: [{deps}]\n"

        prompt += (
            "\nEvaluate if the dependencies logically ensure that prerequisites are met for each step.\n"
            "Does the order make sense to achieve the goal?\n"
            "Respond ONLY with a JSON object:\n"
            "{\n"
            '  "score": 1-10,\n'
            '  "approved": bool,\n'
            '  "feedback": "Reasoning for the score"\n'
            "}"
        )

        messages = [{"role": "system", "content": prompt}]
        try:
            response_text = await self.llm.chat_completion(messages, temperature=0.1)
            # Simple markdown cleaning
            if "```" in response_text:
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"): response_text = response_text[4:]

            return json.loads(response_text.strip())
        except Exception as e:
            logging.error(f"LLM logical evaluation failed: {e}")
            return {"score": 0, "approved": False, "feedback": f"LLM error: {e}"}
