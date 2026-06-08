"""
ACS (Agent Control Specification) Runtime Guard module.
Implements 5 validation checkpoints for agent workflows to standardize runtime guardrails.
"""

import logging
from typing import Dict, Any, Tuple, Optional


class SecurityViolationError(Exception):
    """Exception raised when an action is blocked by the ACS Guard."""
    pass


class ACSGuard:
    """
    Runtime governance layer that intercepts state-changing actions
    and evaluates them through 5 ACS checkpoints before execution.
    """

    def __init__(self) -> None:
        """Initializes the ACS Guard."""
        self.logger = logging.getLogger(__name__)

    def checkpoint_1_input_validation(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 1: Input Validation.
        Validates the raw input data.

        Args:
            workflow_data: The workflow context data.

        Returns:
            A tuple (passed, reason).
        """
        if not workflow_data:
            return False, "Input validation failed: workflow data is empty."
        if "action" not in workflow_data:
            return False, "Input validation failed: missing 'action' field."
        return True, "Input validation passed."

    def checkpoint_2_intent_authorization(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 2: Intent Authorization.
        Verifies if the agent's intent is authorized.

        Args:
            workflow_data: The workflow context data.

        Returns:
            A tuple (passed, reason).
        """
        action = workflow_data.get("action")
        if action == "unauthorized_action":
            return False, f"Intent authorization failed: action '{action}' is not allowed."
        return True, "Intent authorization passed."

    def checkpoint_3_tool_policy(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 3: Tool Policy.
        Checks if the tool complies with defined policies.

        Args:
            workflow_data: The workflow context data.

        Returns:
            A tuple (passed, reason).
        """
        tool = workflow_data.get("tool")
        if tool == "forbidden_tool":
            return False, f"Tool policy failed: tool '{tool}' is forbidden."
        return True, "Tool policy passed."

    def checkpoint_4_state_transition(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 4: State Transition.
        Ensures the proposed state transition is valid.

        Args:
            workflow_data: The workflow context data.

        Returns:
            A tuple (passed, reason).
        """
        current_state = workflow_data.get("current_state")
        next_state = workflow_data.get("next_state")
        if current_state == "error" and next_state == "executing":
            return False, f"State transition failed: cannot transition from '{current_state}' to '{next_state}'."
        return True, "State transition passed."

    def checkpoint_5_output_sanitization(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 5: Output Sanitization.
        Sanitizes the final output.

        Args:
            workflow_data: The workflow context data.

        Returns:
            A tuple (passed, reason).
        """
        output = workflow_data.get("output", "")
        if "secret_key" in str(output):
            return False, "Output sanitization failed: sensitive data detected in output."
        return True, "Output sanitization passed."

    def validate_workflow(self, workflow_data: Dict[str, Any]) -> bool:
        """
        Validates workflow data through all 5 ACS checkpoints.

        Args:
            workflow_data: The workflow context data.

        Returns:
            True if passed, False otherwise.
        """
        checkpoints = [
            self.checkpoint_1_input_validation,
            self.checkpoint_2_intent_authorization,
            self.checkpoint_3_tool_policy,
            self.checkpoint_4_state_transition,
            self.checkpoint_5_output_sanitization
        ]

        for i, checkpoint in enumerate(checkpoints, 1):
            passed, reason = checkpoint(workflow_data)
            if not passed:
                self.logger.warning(f"ACS Checkpoint {i} Failed: {reason}")
                return False
            self.logger.info(f"ACS Checkpoint {i} Passed: {reason}")

        return True

    def intercept_action(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intercepts a state-changing action, validates it, and raises an exception if invalid.

        Args:
            workflow_data: The data for the action to validate.

        Returns:
            The unmodified workflow data if it passed validation.

        Raises:
            SecurityViolationError: If validation fails at any checkpoint.
        """
        checkpoints = [
            self.checkpoint_1_input_validation,
            self.checkpoint_2_intent_authorization,
            self.checkpoint_3_tool_policy,
            self.checkpoint_4_state_transition,
            self.checkpoint_5_output_sanitization
        ]

        for i, checkpoint in enumerate(checkpoints, 1):
            passed, reason = checkpoint(workflow_data)
            if not passed:
                self.logger.warning(f"ACS Checkpoint {i} Failed: {reason}")
                raise SecurityViolationError(f"Action blocked by ACS checkpoint {i}: {reason}")
            self.logger.debug(f"ACS Checkpoint {i} Passed: {reason}")

        return workflow_data
