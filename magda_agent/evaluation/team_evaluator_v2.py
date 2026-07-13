import json
import logging
from typing import Dict, Any, List, Optional
from magda_agent.llm_client import LLMClient
from magda_agent.evaluation.evaluator_dependency_v3 import EvaluatorDependencyV3

class TeamEvaluatorV2:
    """
    TeamEvaluatorV2 evaluates the execution results of a multi-agent team.
    It combines structural dependency validation with holistic LLM-based output review.
    Inspired by Claude Agent SDK Agent Teams trend (June 2026).
    """
    def __init__(self, llm: LLMClient) -> None:
        """
        Initializes the TeamEvaluatorV2.

        Args:
            llm: The LLM client for cognitive evaluation.
        """
        self.llm = llm
        self.dependency_evaluator = EvaluatorDependencyV3(llm=llm)

    async def evaluate_team_execution(
        self,
        plan: Dict[str, Any],
        results: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Evaluates the combined results of a team execution against the original plan.

        Args:
            plan (Dict[str, Any]): The original execution plan (with 'steps' and 'goal').
            results (Dict[str, str]): A mapping of step IDs to their execution results.

        Returns:
            Dict[str, Any]: Evaluation result containing score, approved status, and feedback.
        """
        logging.info("Starting TeamEvaluatorV2 multi-agent execution review...")

        # 1. Structural Dependency Validation
        dep_eval = await self.dependency_evaluator.evaluate_plan_dependencies(plan)
        if not dep_eval.get("approved", False):
            return {
                "score": dep_eval.get("score", 0),
                "approved": False,
                "feedback": f"Structural validation failed: {dep_eval.get('feedback')}",
                "metadata": {"dep_eval": dep_eval}
            }

        # 2. Completeness Check
        steps = plan.get("steps", [])
        missing_results = []
        for step in steps:
            step_id = step.get("id")
            if step_id and step_id not in results:
                missing_results.append(step_id)

        if missing_results:
            return {
                "score": 0,
                "approved": False,
                "feedback": f"Execution incomplete. Missing results for steps: {missing_results}",
                "metadata": {"dep_eval": dep_eval, "missing_steps": missing_results}
            }

        # 3. Holistic LLM Review of Combined Outputs
        holistic_eval = await self._holistic_review(plan, results)

        final_approved = holistic_eval.get("approved", False)

        return {
            "score": holistic_eval.get("score", 0) if final_approved else min(holistic_eval.get("score", 0), 5),
            "approved": final_approved,
            "feedback": holistic_eval.get("feedback", ""),
            "metadata": {
                "dep_eval": dep_eval,
                "holistic_eval": holistic_eval
            }
        }

    async def _holistic_review(self, plan: Dict[str, Any], results: Dict[str, str]) -> Dict[str, Any]:
        """Uses LLM to perform a holistic review of all subagent outputs against the goal."""
        goal = plan.get("goal", "No goal specified")

        # Combine all results into a single context
        combined_results_str = ""
        for step_id, result in results.items():
            combined_results_str += f"--- Step ID: {step_id} ---\n{result}\n"

        prompt = (
            "You are a Lead Evaluator for an AI Agent Team. Your task is to evaluate if the combined outputs "
            "of several subagents successfully achieve the overarching goal.\n\n"
            f"Overarching Goal: {goal}\n\n"
            "Combined Execution Results:\n"
            f"{combined_results_str}\n"
            "Evaluate the results for consistency, completeness, and accuracy relative to the goal.\n"
            "Respond ONLY with a JSON object:\n"
            "{\n"
            '  "score": 1-10,\n'
            '  "approved": bool,\n'
            '  "feedback": "Detailed reasoning for the evaluation"\n'
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
            logging.error(f"Holistic team review failed: {e}")
            return {"score": 0, "approved": False, "feedback": f"LLM error: {e}"}
