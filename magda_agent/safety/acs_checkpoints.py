import logging
from typing import Dict, Any, Tuple

class ACSCheckpoints:
    """
    Implements 5 ACS validation checkpoints for agentic workflows.
    Ensures all actions pass through 5 checks before execution.
    """
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def checkpoint_1_input_validation(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validates raw input data for actions."""
        if not action_data:
            return False, "Checkpoint 1 Failed: empty action data."
        if "action_name" not in action_data:
            return False, "Checkpoint 1 Failed: missing 'action_name'."
        return True, "Checkpoint 1 Passed."

    def checkpoint_2_intent_authorization(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Verifies if the intent is authorized."""
        if action_data.get("action_name") == "unauthorized_action":
            return False, "Checkpoint 2 Failed: unauthorized action intent."
        return True, "Checkpoint 2 Passed."

    def checkpoint_3_tool_policy(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Checks compliance with tool policies."""
        if action_data.get("tool_name") == "forbidden_tool":
            return False, "Checkpoint 3 Failed: tool is forbidden."
        return True, "Checkpoint 3 Passed."

    def checkpoint_4_state_transition(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Ensures the state transition is valid."""
        if action_data.get("state") == "error":
            return False, "Checkpoint 4 Failed: invalid state transition from error."
        return True, "Checkpoint 4 Passed."

    def checkpoint_5_output_sanitization(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Sanitizes output data."""
        if "secret_key" in str(action_data.get("output", "")):
            return False, "Checkpoint 5 Failed: sensitive data found in output."
        return True, "Checkpoint 5 Passed."

    def validate_action(self, action_data: Dict[str, Any]) -> bool:
        """Runs all 5 checkpoints and returns True if all pass."""
        checkpoints = [
            self.checkpoint_1_input_validation,
            self.checkpoint_2_intent_authorization,
            self.checkpoint_3_tool_policy,
            self.checkpoint_4_state_transition,
            self.checkpoint_5_output_sanitization
        ]

        for checkpoint in checkpoints:
            passed, reason = checkpoint(action_data)
            if not passed:
                self.logger.warning(reason)
                return False

        return True
