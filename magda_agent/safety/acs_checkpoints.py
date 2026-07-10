import logging
import re
from typing import Dict, Any, Tuple, Optional
from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.audit_trail import AuditTrail

_SENSITIVE_PATTERNS = (
    re.compile(r"api[_-]?key|token|password|private[_-]?key|secret[_-]?key", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
    re.compile(r"\.env", re.IGNORECASE),
    re.compile(r"secrets?", re.IGNORECASE),
)

class SecurityViolationError(Exception):
    """Exception raised when an action is blocked by ACS checkpoints."""
    pass

class ACSCheckpoints:
    """
    Implements 5 ACS validation checkpoints for agentic workflows.
    Ensures all actions pass through 5 checks before execution.
    """
    def __init__(self, policy_layer: Optional[PolicyLayer] = None, audit_trail: Optional[AuditTrail] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.policy_layer = policy_layer or PolicyLayer()
        self.audit_trail = audit_trail or AuditTrail()

    def checkpoint_1_input_validation(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Checkpoint 1: Input Validation. Ensures action data is well-formed."""
        if not isinstance(action_data, dict):
            return False, "Checkpoint 1 Failed: workflow data must be a dictionary."
        if not action_data:
            return False, "Checkpoint 1 Failed: empty action data."

        required_fields = ["action_name", "tool_name"]
        for field in required_fields:
            if field not in action_data:
                return False, f"Checkpoint 1 Failed: missing '{field}'."
            if not isinstance(action_data[field], str):
                return False, f"Checkpoint 1 Failed: '{field}' must be a string."

        return True, "Checkpoint 1 Passed."

    def checkpoint_2_intent_authorization(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Checkpoint 2: Intent Authorization. Verifies if the intent is authorized."""
        action = action_data.get("action_name")
        allowed_intents = {
            "read", "write", "execute", "plan", "reflect", "delegate", "analyze", "chat", "test_action"
        }
        if action == "unauthorized_action":
            return False, f"Checkpoint 2 Failed: action '{action}' is explicitly blacklisted."
        if action not in allowed_intents:
            return False, f"Checkpoint 2 Failed: action '{action}' is not in allowed intents."
        return True, "Checkpoint 2 Passed."

    def checkpoint_3_tool_policy(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Checkpoint 3: Tool Policy. Checks compliance with tool policies."""
        tool = action_data.get("tool_name")
        if tool == "forbidden_tool":
            return False, f"Checkpoint 3 Failed: tool '{tool}' is forbidden."

        kwargs = action_data.get("kwargs", {})
        allow, explanation = self.policy_layer.evaluate(tool, **kwargs)
        if not allow:
            return False, f"Checkpoint 3 Failed: {explanation}"

        return True, "Checkpoint 3 Passed."

    def checkpoint_4_state_transition(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Checkpoint 4: State Transition. Ensures the state transition is valid."""
        current_state = action_data.get("state", "idle")
        next_state = action_data.get("next_state")

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

        if not next_state:
            return True, "Checkpoint 4 Passed: next_state not provided."

        if next_state not in allowed_transitions[current_state] and next_state != "error":
            return False, f"Checkpoint 4 Failed: cannot transition from '{current_state}' to '{next_state}'."

        return True, "Checkpoint 4 Passed."

    def checkpoint_5_output_sanitization(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Checkpoint 5: Output Sanitization. Sanitizes output data."""
        output = action_data.get("output")
        if output is None:
            return True, "Checkpoint 5 Passed: no output to sanitize."

        output_str = str(output)
        for pattern in _SENSITIVE_PATTERNS:
            if pattern.search(output_str):
                return False, f"Checkpoint 5 Failed: sensitive pattern '{pattern.pattern}' detected in output."

        return True, "Checkpoint 5 Passed."

    def validate_pre_execution(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Runs checkpoints 1 to 4 and logs to audit trail on failure."""
        for i, cp in enumerate([
            self.checkpoint_1_input_validation,
            self.checkpoint_2_intent_authorization,
            self.checkpoint_3_tool_policy,
            self.checkpoint_4_state_transition
        ], 1):
            ok, reason = cp(action_data)
            if not ok:
                self.logger.warning(reason)
                self.audit_trail.log_call(
                    tool_name=action_data.get("tool_name", "unknown"),
                    kwargs=action_data.get("kwargs", {}),
                    why=reason,
                    result="blocked",
                    duration=0.0
                )
                return False, reason
            self.logger.debug(reason)
        return True, "Pre-execution ACS checkpoints passed."

    def validate_post_execution(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Runs checkpoint 5 and logs to audit trail on failure."""
        ok, reason = self.checkpoint_5_output_sanitization(action_data)
        if not ok:
            self.logger.warning(reason)
            self.audit_trail.log_call(
                tool_name=action_data.get("tool_name", "unknown"),
                kwargs=action_data.get("kwargs", {}),
                why=reason,
                result="blocked",
                duration=0.0
            )
            return False, reason
        self.logger.debug(reason)
        return True, reason

    def validate_action(self, action_data: Dict[str, Any]) -> bool:
        """Runs all 5 checkpoints and returns True if all pass."""
        ok_pre, reason_pre = self.validate_pre_execution(action_data)
        if not ok_pre:
            return False

        ok_post, reason_post = self.validate_post_execution(action_data)
        if not ok_post:
            return False

        self.audit_trail.log_call(
            tool_name=action_data.get("tool_name", "unknown"),
            kwargs=action_data.get("kwargs", {}),
            why="All 5 ACS checkpoints passed.",
            result="allowed",
            duration=0.0
        )
        return True

    def intercept_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intercepts an action, validates it, and raises SecurityViolationError on failure.
        Useful for synchronous middleware-like validation.
        """
        if not self.validate_action(action_data):
            # The reason is already logged in validate_action, but we need it for the exception.
            # Re-running to get the exact failure if necessary, or just a generic error.
            # For efficiency in a real system we'd return (bool, str) from validate_action.
            # Let's just find the failing one.
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
                     raise SecurityViolationError(reason)
        return action_data
