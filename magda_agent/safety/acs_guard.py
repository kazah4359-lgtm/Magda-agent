"""
ACS (Agent Control Specification) Runtime Guard module.
Implements 5 validation checkpoints for agent workflows to standardize runtime guardrails.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List


class SecurityViolationError(Exception):
    """Exception raised when an action is blocked by the ACS Guard."""
    pass


class ACSCheckpoint(ABC):
    """Abstract base class for ACS validation checkpoints."""

    @abstractmethod
    def validate(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates the workflow data against this checkpoint.

        Args:
            workflow_data: The workflow context data.

        Returns:
            A tuple (passed, reason).
        """
        pass


class InputValidationCheckpoint(ACSCheckpoint):
    """
    Checkpoint 1: Input Validation.
    Validates the raw input data.
    """

    def validate(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        if not workflow_data:
            return False, "Input validation failed: workflow data is empty."
        if "action" not in workflow_data:
            return False, "Input validation failed: missing 'action' field."
        return True, "Input validation passed."


class IntentAuthorizationCheckpoint(ACSCheckpoint):
    """
    Checkpoint 2: Intent Authorization.
    Verifies if the agent's intent is authorized.
    """

    def validate(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        action = workflow_data.get("action")
        if action == "unauthorized_action":
            return False, f"Intent authorization failed: action '{action}' is not allowed."
        return True, "Intent authorization passed."


class ToolPolicyCheckpoint(ACSCheckpoint):
    """
    Checkpoint 3: Tool Policy.
    Checks if the tool complies with defined policies.
    """

    def validate(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        tool = workflow_data.get("tool")
        if tool == "forbidden_tool":
            return False, f"Tool policy failed: tool '{tool}' is forbidden."
        return True, "Tool policy passed."


class StateTransitionCheckpoint(ACSCheckpoint):
    """
    Checkpoint 4: State Transition.
    Ensures the proposed state transition is valid.
    """

    def validate(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        current_state = workflow_data.get("current_state")
        next_state = workflow_data.get("next_state")
        if current_state == "error" and next_state == "executing":
            return False, f"State transition failed: cannot transition from '{current_state}' to '{next_state}'."
        return True, "State transition passed."


class OutputSanitizationCheckpoint(ACSCheckpoint):
    """
    Checkpoint 5: Output Sanitization.
    Sanitizes the final output.
    """

    def validate(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        output = workflow_data.get("output", "")
        if "secret_key" in str(output):
            return False, "Output sanitization failed: sensitive data detected in output."
        return True, "Output sanitization passed."


class ACSGuard:
    """
    Runtime governance layer that intercepts state-changing actions
    and evaluates them through 5 ACS checkpoints before execution.
    """

    def __init__(self) -> None:
        """Initializes the ACS Guard."""
        self.logger = logging.getLogger(__name__)
        self.checkpoints: List[ACSCheckpoint] = [
            InputValidationCheckpoint(),
            IntentAuthorizationCheckpoint(),
            ToolPolicyCheckpoint(),
            StateTransitionCheckpoint(),
            OutputSanitizationCheckpoint()
        ]

    def validate_workflow(self, workflow_data: Dict[str, Any]) -> bool:
        """
        Validates workflow data through all 5 ACS checkpoints.

        Args:
            workflow_data: The workflow context data.

        Returns:
            True if passed, False otherwise.
        """
        for i, checkpoint in enumerate(self.checkpoints, 1):
            passed, reason = checkpoint.validate(workflow_data)
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
        for i, checkpoint in enumerate(self.checkpoints, 1):
            passed, reason = checkpoint.validate(workflow_data)
            if not passed:
                self.logger.warning(f"ACS Checkpoint {i} Failed: {reason}")
                raise SecurityViolationError(f"Action blocked by ACS checkpoint {i}: {reason}")
            self.logger.debug(f"ACS Checkpoint {i} Passed: {reason}")

        return workflow_data
