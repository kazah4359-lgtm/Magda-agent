import json
import logging
from typing import Dict, Any, List

from magda_agent.llm_client import LLMClient

class AssertFramework:
    """
    ASSERT Policy Driven Evaluation Framework.
    Evaluates an execution plan against system constraints dynamically.
    """

    def __init__(self, llm: LLMClient) -> None:
        """
        Initializes the AssertFramework.

        Args:
            llm: The LLM client to use for evaluation.
        """
        self.llm = llm

    async def evaluate_plan(self, plan: List[str], policies: List[str]) -> Dict[str, Any]:
        """
        Evaluates a given execution plan against explicit system policies.

        Args:
            plan: A list of strings representing the steps of the execution plan.
            policies: A list of string policies to check against.

        Returns:
            A dictionary containing 'is_compliant' (bool) and 'violations' (List[str]).
        """
        formatted_plan = "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan)])
        formatted_policies = "\n".join([f"- {policy}" for policy in policies])

        prompt = (
            "Evaluate the following execution plan against the provided system policies.\n"
            "Determine if the plan violates ANY of these policies.\n\n"
            f"Policies:\n{formatted_policies}\n\n"
            f"Execution Plan:\n{formatted_plan}\n\n"
            "Respond ONLY with a JSON object in this exact format:\n"
            "{\n"
            '  "is_compliant": true,\n'
            '  "violations": ["Policy description if violated, else empty"]\n'
            "}"
        )

        messages = [{"role": "system", "content": prompt}]
        max_retries = 3

        for attempt in range(max_retries):
            try:
                evaluation_text = await self.llm.chat_completion(messages, temperature=0.1)

                if "```" in evaluation_text:
                    evaluation_text = evaluation_text.split("```")[1]
                    if evaluation_text.startswith("json"):
                        evaluation_text = evaluation_text[4:]

                evaluation = json.loads(evaluation_text.strip())
                is_compliant = evaluation.get("is_compliant", False)
                violations = evaluation.get("violations", [])

                if not is_compliant and violations:
                    logging.warning(f"ASSERT Framework: Plan violation detected! Violations: {violations}")

                return {
                    "is_compliant": is_compliant,
                    "violations": violations
                }

            except json.JSONDecodeError as e:
                logging.warning(f"ASSERT Framework JSON decoding error attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    logging.error("ASSERT Framework reached max retries. Failing closed (is_compliant=False).")
                    return {"is_compliant": False, "violations": ["Evaluation failed due to JSON decoding error."]}
            except Exception as e:
                logging.error(f"ASSERT Framework failed: {e}")
                return {"is_compliant": False, "violations": [f"Evaluation failed: {str(e)}"]}

        return {"is_compliant": False, "violations": ["Unknown error"]}
