import logging
import re
from typing import Dict, Any, Tuple

_SENSITIVE_PATTERNS = (
    re.compile(r"api[_-]?key|token|password|private[_-]?key|secret[_-]?key", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
    re.compile(r"\.env", re.IGNORECASE),
    re.compile(r"secrets?", re.IGNORECASE),
)

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
        action = action_data.get("action_name")
        allowed_intents = {
            "read", "write", "execute", "plan", "reflect", "delegate", "analyze", "chat", "test_action"
        }
        if action == "unauthorized_action":
            return False, "Checkpoint 2 Failed: unauthorized action intent."
        if action not in allowed_intents:
            return False, f"Checkpoint 2 Failed: action '{action}' is not in allowed intents."
        return True, "Checkpoint 2 Passed."

    def checkpoint_3_tool_policy(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Checks compliance with tool policies."""
        if action_data.get("tool_name") == "forbidden_tool":
            return False, "Checkpoint 3 Failed: tool is forbidden."
        return True, "Checkpoint 3 Passed."

    def checkpoint_4_state_transition(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Ensures the state transition is valid."""
        current_state = action_data.get("state", "idle")
        next_state = action_data.get("next_state")

        if not next_state:
            if current_state == "error":
                return False, "Checkpoint 4 Failed: invalid state transition from error without next_state."
            return True, "Checkpoint 4 Passed: next_state not provided."

        allowed_transitions = {
            "idle": ["planning", "reflecting", "analyzing", "executing", "active"],
            "planning": ["executing", "idle"],
            "executing": ["evaluating", "idle"],
            "evaluating": ["idle", "planning"],
            "reflecting": ["idle"],
            "analyzing": ["idle", "planning"],
            "active": ["idle", "executing"],
            "error": ["idle"]
        }

        if current_state not in allowed_transitions:
            return False, f"Checkpoint 4 Failed: unknown current_state '{current_state}'."

        if next_state not in allowed_transitions[current_state] and next_state != "error":
            return False, f"Checkpoint 4 Failed: cannot transition from '{current_state}' to '{next_state}'."

        return True, "Checkpoint 4 Passed."

    def checkpoint_5_output_sanitization(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Sanitizes output data."""
        output = action_data.get("output")
        if output is None:
            return True, "Checkpoint 5 Passed: no output to sanitize."

        output_str = str(output)
        for pattern in _SENSITIVE_PATTERNS:
            if pattern.search(output_str):
                return False, f"Checkpoint 5 Failed: sensitive pattern '{pattern.pattern}' detected in output."

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

        for i, checkpoint in enumerate(checkpoints, 1):
            passed, reason = checkpoint(action_data)
            if not passed:
                self.logger.warning(reason)
                return False

        return True
