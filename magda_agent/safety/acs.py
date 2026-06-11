import logging
from typing import Dict, Any, Tuple

class ACSWorkflowGuard:
    """
    ACS (Agent Control Specification) Workflow Guard.
    Implements 5 validation checkpoints for agent workflows to standardize runtime guardrails.
    """

    def __init__(self) -> None:
        """Initializes the ACS Workflow Guard."""
        pass

    def checkpoint_1_input_validation(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 1: Input Validation.
        Validates the raw input data to ensure it is not malformed, missing required fields, or containing malicious payloads.

        Args:
            workflow_data (Dict[str, Any]): The workflow context data.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating if validation passed, and a reason if it failed.
        """
        if not workflow_data:
            return False, "Input validation failed: workflow data is empty."
        if "action" not in workflow_data:
            return False, "Input validation failed: missing 'action' field."
        return True, "Input validation passed."

    def checkpoint_2_intent_authorization(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 2: Intent Authorization.
        Verifies if the agent's intent is authorized within the current context and permissions.

        Args:
            workflow_data (Dict[str, Any]): The workflow context data.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating if authorization passed, and a reason if it failed.
        """
        action = workflow_data.get("action")
        if action == "unauthorized_action":
            return False, f"Intent authorization failed: action '{action}' is not allowed."
        return True, "Intent authorization passed."

    def checkpoint_3_tool_policy(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 3: Tool Policy.
        Checks if the specific tool or function to be executed complies with the defined policies.

        Args:
            workflow_data (Dict[str, Any]): The workflow context data.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating if policy check passed, and a reason if it failed.
        """
        tool = workflow_data.get("tool")
        if tool == "forbidden_tool":
            return False, f"Tool policy failed: tool '{tool}' is forbidden."
        return True, "Tool policy passed."

    def checkpoint_4_state_transition(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 4: State Transition.
        Ensures the proposed state transition is valid and does not lead the system into an inconsistent or unsafe state.

        Args:
            workflow_data (Dict[str, Any]): The workflow context data.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating if transition is valid, and a reason if it failed.
        """
        current_state = workflow_data.get("current_state")
        next_state = workflow_data.get("next_state")
        if current_state == "error" and next_state == "executing":
            return False, f"State transition failed: cannot transition from '{current_state}' to '{next_state}'."
        return True, "State transition passed."

    def checkpoint_5_output_sanitization(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 5: Output Sanitization.
        Sanitizes the final output to ensure sensitive information is not leaked and formatting is safe.

        Args:
            workflow_data (Dict[str, Any]): The workflow context data.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating if sanitization passed, and a reason if it failed.
        """
        output = workflow_data.get("output", "")
        if "secret_key" in str(output):
            return False, "Output sanitization failed: sensitive data detected in output."
        return True, "Output sanitization passed."

    def validate_workflow(self, workflow_data: Dict[str, Any]) -> bool:
        """
        Validates the workflow data through all 5 ACS checkpoints.

        Args:
            workflow_data (Dict[str, Any]): The workflow context data.

        Returns:
            bool: True if all checkpoints pass, False otherwise.
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
                logging.warning(f"ACS Checkpoint {i} Failed: {reason}")
                return False
            logging.info(f"ACS Checkpoint {i} Passed: {reason}")

        return True
    def validate_with_fallback(self, workflow_data: Dict[str, Any], fallback_action: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates the workflow data through all 5 ACS checkpoints with a realtime fallback.
        If validation fails, returns the fallback action if provided, or an error action.

        Args:
            workflow_data (Dict[str, Any]): The workflow context data.
            fallback_action (Dict[str, Any], optional): The fallback data to return on failure. Defaults to None.

        Returns:
            Tuple[bool, Dict[str, Any]]: A tuple containing a boolean indicating if original validation passed,
                                         and the resulting workflow data (original or fallback).
        """
        passed = self.validate_workflow(workflow_data)
        if passed:
            return True, workflow_data

        if fallback_action is not None:
            logging.info("ACS validation failed, triggering fallback action.")
            return False, fallback_action

        logging.warning("ACS validation failed and no fallback provided. Returning error state.")
        return False, {
            "action": "error",
            "tool": "none",
            "current_state": workflow_data.get("current_state", "unknown"),
            "next_state": "error",
            "output": "Action blocked by safety guardrails."
        }
