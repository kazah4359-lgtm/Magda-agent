import json
import logging
from typing import Dict, Any, List, Optional
from magda_agent.llm_client import LLMClient

class EvaluatorReflectionV3:
    """
    EvaluatorReflectionV3 is a Claude-inspired specialized evaluation module.
    It analyzes the independent execution logs/traces of isolated subagents
    and generates reflection metrics and actionable feedback for the Planner.
    """

    def __init__(self, llm: LLMClient) -> None:
        """
        Initializes the EvaluatorReflectionV3 with an LLMClient.

        Args:
            llm (LLMClient): The Language Model client to use for evaluations.
        """
        self.llm = llm

    def _parse_and_calculate_metrics(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parses subagent logs and generates structural metrics.

        Args:
            logs (List[Dict[str, Any]]): A list of dicts representing subagent execution traces.

        Returns:
            Dict[str, Any]: Calculated metrics such as success rate, total time, and errors.
        """
        if not logs:
            return {
                "total_subagents": 0,
                "successful_subagents": 0,
                "failed_subagents": 0,
                "success_rate": 0.0,
                "total_execution_time": 0.0,
                "total_errors": 0,
                "error_density": 0.0,
            }

        total_subagents = len(logs)
        successful_subagents = 0
        total_time = 0.0
        total_errors = 0

        for log in logs:
            status = log.get("status", "success")
            if status == "success":
                successful_subagents += 1

            total_time += float(log.get("execution_time_seconds", 0.0))

            errors = log.get("errors", [])
            total_errors += len(errors)

        success_rate = successful_subagents / total_subagents if total_subagents > 0 else 0.0
        error_density = total_errors / total_subagents if total_subagents > 0 else 0.0

        return {
            "total_subagents": total_subagents,
            "successful_subagents": successful_subagents,
            "failed_subagents": total_subagents - successful_subagents,
            "success_rate": success_rate,
            "total_execution_time": total_time,
            "total_errors": total_errors,
            "error_density": error_density,
        }

    async def reflect_on_subagent_logs(
        self,
        logs: List[Dict[str, Any]],
        planner_goal: str
    ) -> Dict[str, Any]:
        """
        Analyzes subagent logs, calculates metrics, and uses the LLM to generate cognitive reflections.

        Args:
            logs (List[Dict[str, Any]]): A list of execution traces (logs) from parallel subagents.
            planner_goal (str): The overarching goal the planner set for the subagents.

        Returns:
            Dict[str, Any]: A structured reflection dictionary containing:
                - metrics: Python-calculated performance metrics.
                - reflection: LLM-generated reflection, critique, and feedback for the planner.
                - approved: Boolean indicating if the overall execution meets quality standards.
        """
        logging.info("Starting EvaluatorReflectionV3 on subagent logs...")

        # Calculate Python metrics
        calculated_metrics = self._parse_and_calculate_metrics(logs)

        # Build prompt for LLM cognitive reflection
        logs_summary = ""
        for i, log in enumerate(logs):
            subagent_id = log.get("subagent_id", f"subagent_{i}")
            task = log.get("task", "No task description")
            status = log.get("status", "unknown")
            output = log.get("output", "")
            system_logs = "\n".join(log.get("system_logs", []))
            errors = "\n".join(log.get("errors", []))

            logs_summary += (
                f"--- Subagent ID: {subagent_id} ---\n"
                f"Task: {task}\n"
                f"Status: {status}\n"
                f"Output: {output}\n"
                f"System Logs:\n{system_logs}\n"
                f"Errors:\n{errors}\n\n"
            )

        prompt = (
            "You are an expert Evaluator Agent. Your task is to analyze and reflect on the independent execution "
            "logs of isolated subagents that were spawned to achieve an overarching goal.\n\n"
            f"Overarching Goal:\n{planner_goal}\n\n"
            "Subagent Execution Logs & Traces:\n"
            f"{logs_summary}"
            "Please perform a deep evaluation. "
            "Respond ONLY with a JSON object containing the following keys:\n"
            "{\n"
            '  "approved": bool,\n'
            '  "quality_score": int,\n'
            '  "critique": "Detailed critique of the subagents\' outputs relative to the goal",\n'
            '  "bottlenecks": ["List of subagents or tasks that caused issues, failures, or bottlenecks"],\n'
            '  "actionable_feedback_for_planner": "Concrete feedback on how the Planner can adjust or break down the tasks to improve future execution"\n'
            "}"
        )

        messages = [
            {"role": "system", "content": "You are a precise and critical AI evaluation agent. Output JSON only."},
            {"role": "user", "content": prompt}
        ]

        try:
            response_text = await self.llm.chat_completion(messages, temperature=0.1)

            # Markdown code-block cleanup
            if "```" in response_text:
                response_text = response_text.split("```")[1]
                if response_text.lower().startswith("json"):
                    response_text = response_text[4:]

            llm_reflection = json.loads(response_text.strip())

            # Enforce schema validity
            required_keys = {"approved", "quality_score", "critique", "bottlenecks", "actionable_feedback_for_planner"}
            if not required_keys.issubset(llm_reflection.keys()):
                raise ValueError("LLM response lacks required JSON keys.")

        except Exception as e:
            logging.error(f"LLM reflection parsing failed: {e}")
            llm_reflection = {
                "approved": calculated_metrics["success_rate"] >= 1.0,
                "quality_score": int(calculated_metrics["success_rate"] * 10),
                "critique": f"Failed to perform LLM analysis due to error: {e}",
                "bottlenecks": [log.get("subagent_id", "unknown") for log in logs if log.get("status") != "success"],
                "actionable_feedback_for_planner": "Please inspect subagent error logs directly."
            }

        return {
            "metrics": calculated_metrics,
            "reflection": llm_reflection,
            "approved": llm_reflection.get("approved", False) and calculated_metrics["success_rate"] > 0.0
        }
