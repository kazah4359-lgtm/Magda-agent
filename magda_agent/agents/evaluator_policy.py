import logging
from typing import Dict, Any, List, Optional, Protocol

class EvaluatorProtocol(Protocol):
    """
    Protocol defining the expected interface for an evaluator sub-agent.
    """
    async def evaluate_generator_output(self, generator_output: str, user_request: str) -> Dict[str, Any]:
        ...

class PolicyEnforcedEvaluator:
    """
    Wraps an Evaluator agent to strictly enforce system policies on the generator's output.
    Inspired by Claude Agent SDK Subagent policy patterns.
    """
    def __init__(self, evaluator_agent: EvaluatorProtocol):
        """
        Initializes the PolicyEnforcedEvaluator.

        Args:
            evaluator_agent (EvaluatorProtocol): The underlying evaluator sub-agent (e.g., EvaluatorAgentV3).
        """
        self.evaluator = evaluator_agent

    async def evaluate_generator_output(
        self,
        generator_output: str,
        user_request: str,
        policies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates the output produced by the Generator Agent and enforces strict system constraints.

        Args:
            generator_output (str): The text or code output generated.
            user_request (str): The initial user request or context.
            policies (Optional[List[str]]): A list of strict policies to enforce.
                                            Format: 'forbidden:<string>', 'must_contain:<string>', or 'max_length:<int>'.

        Returns:
            Dict[str, Any]: Parsed JSON representing the critique and feedback containing keys:
                - score (int): 1-10 evaluation score.
                - approved (bool): Whether the output is acceptable and passes all policies.
                - feedback (str): Detailed, actionable critique including any policy violations.
        """
        policies = policies or []

        # 1. Enforce strict local policies first
        violations = self._check_policies(generator_output, policies)
        if violations:
            logging.warning(f"Output rejected due to policy violations: {violations}")
            return {
                "score": 0,
                "approved": False,
                "feedback": f"Policy Violation: Output failed to comply with strict system constraints. Violations: {', '.join(violations)}"
            }

        # 2. Delegate to the underlying LLM evaluator
        try:
            return await self.evaluator.evaluate_generator_output(generator_output, user_request)
        except Exception as e:
            logging.error(f"Underlying evaluation failed: {e}")
            return {
                "score": 0,
                "approved": False,
                "feedback": f"Evaluation error: {str(e)}"
            }

    def _check_policies(self, output: str, policies: List[str]) -> List[str]:
        """
        Checks the output against the specified strict policies.

        Args:
            output (str): The generator output.
            policies (List[str]): List of policy strings.

        Returns:
            List[str]: A list of policy violation messages. Empty if no violations.
        """
        violations = []
        for policy in policies:
            policy_lower = policy.lower()
            if policy_lower.startswith("forbidden:"):
                term = policy_lower[len("forbidden:"):].strip()
                if term in output.lower():
                    violations.append(f"Contains forbidden term: '{term}'")
            elif policy_lower.startswith("must_contain:"):
                term = policy_lower[len("must_contain:"):].strip()
                if term not in output.lower():
                    violations.append(f"Missing required term: '{term}'")
            elif policy_lower.startswith("max_length:"):
                try:
                    max_len = int(policy_lower[len("max_length:"):].strip())
                    if len(output) > max_len:
                        violations.append(f"Exceeds maximum length of {max_len} characters")
                except ValueError:
                    logging.warning(f"Invalid max_length policy: {policy}")
            else:
                logging.debug(f"Unsupported strict policy format ignored: {policy}")

        return violations
