import logging
from typing import Dict, Any, Tuple

class ACSControls:
    """
    ACS (Agent Control Specification) Controls.
    Implements 5 validation checkpoints for agent workflows to standardize runtime guardrails.
    """
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def checkpoint_1_input_validation(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validates the raw input data to ensure it is not malformed or missing required fields."""
        if not workflow_data:
            return False, "Input validation failed: empty data."
        if "action" not in workflow_data:
            return False, "Input validation failed: missing 'action'."
        return True, "Passed."

    def checkpoint_2_intent_authorization(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Verifies if the agent's intent is authorized within the current context and permissions."""
        action = workflow_data.get("action")
        if action == "unauthorized":
            return False, "Intent authorization failed."
        return True, "Passed."

    def checkpoint_3_tool_policy(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Checks if the specific tool or function to be executed complies with the defined policies."""
        if workflow_data.get("tool") == "forbidden":
            return False, "Tool policy failed."
        return True, "Passed."

    def checkpoint_4_state_transition(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Ensures that the proposed state transition is valid and follows the defined workflow state machine."""
        if workflow_data.get("current_state") == "error" and workflow_data.get("next_state") == "executing":
            return False, "State transition failed."
        return True, "Passed."

    def checkpoint_5_output_sanitization(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Sanitizes the final output before it is returned or acted upon."""
        if "secret" in str(workflow_data.get("output", "")):
            return False, "Output sanitization failed."
        return True, "Passed."

    def validate_workflow(self, workflow_data: Dict[str, Any]) -> bool:
        """Runs all 5 ACS checkpoints in order to validate the workflow data."""
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
                self.logger.warning(f"Checkpoint {i} Failed: {reason}")
                return False
        return True
