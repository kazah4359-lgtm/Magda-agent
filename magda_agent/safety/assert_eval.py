import json
import logging
from typing import Dict, Any, List, Optional

from magda_agent.llm_client import LLMClient

class AssertWorkflowEvaluator:
    """
    ASSERT Policy-Driven Evaluation Framework for Agentic Workflows.
    Evaluates whether a given workflow step or data violates explicit security/safety policies.
    """

    def __init__(self, llm: LLMClient) -> None:
        """
        Initializes the AssertWorkflowEvaluator.

        Args:
            llm: The LLM client to use for evaluation.
        """
        self.llm = llm

    async def evaluate_workflow(self, workflow_data: Dict[str, Any], policies: List[str]) -> bool:
        """
        Evaluates a workflow against a set of policies using the LLM.

        Args:
            workflow_data: The data describing the workflow action or state.
            policies: A list of explicit policy strings.

        Returns:
            True if the workflow COMPLIES with all policies (safe), False if it VIOLATES any policy (unsafe).
        """
        formatted_policies = "\n".join([f"- {policy}" for policy in policies])
        workflow_json = json.dumps(workflow_data, indent=2)

        prompt = (
            "Evaluate the following workflow data against the provided security/safety policies.\n"
            "Determine if the workflow action violates ANY of these policies.\n\n"
            f"Policies:\n{formatted_policies}\n\n"
            f"Workflow Data:\n{workflow_json}\n\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            "{\n"
            '  "is_safe": true,\n'
            '  "violated_policies": ["Policy description if violated, else empty string"],\n'
            '  "reason": "Explanation for the decision"\n'
            "}"
        )

        messages = [{"role": "system", "content": prompt}]
        max_retries = 3

        for attempt in range(max_retries):
            try:
                evaluation_text = await self.llm.chat_completion(messages, temperature=0.1)

                # Clean up markdown if present
                if "```" in evaluation_text:
                    evaluation_text = evaluation_text.split("```")[1]
                    if evaluation_text.startswith("json"):
                        evaluation_text = evaluation_text[4:]

                evaluation = json.loads(evaluation_text.strip())
                is_safe = evaluation.get("is_safe", False)

                if not is_safe:
                    violations = evaluation.get("violated_policies", [])
                    reason = evaluation.get("reason", "No reason provided")
                    logging.warning(f"ASSERT Workflow Evaluator: Policy violation detected! Reason: {reason}. Violations: {violations}")

                return is_safe

            except json.JSONDecodeError as e:
                logging.warning(f"ASSERT JSON decoding error attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    logging.error("ASSERT Evaluator reached max retries. Failing closed (safe=False).")
                    return False
            except Exception as e:
                logging.error(f"ASSERT Evaluator failed: {e}")
                return False

        return False
