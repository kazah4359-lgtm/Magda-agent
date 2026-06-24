import logging
import re
from typing import Dict, Any, Tuple, List, Optional
from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.fallback import RealtimeGuardrailFallback

_SENSITIVE_PATTERNS = (
    re.compile(r"api[_-]?key|token|password|private[_-]?key", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
)

class SecurityViolationError(Exception):
    """Exception raised when an action is blocked by the ACS Guard."""
    pass

class ACSWorkflowGuard:
    """
    ACS (Agent Control Specification) Workflow Guard.
    Implements 5 validation checkpoints for agent workflows to standardize runtime guardrails.
    """

    def __init__(self, policy_layer: Optional[PolicyLayer] = None) -> None:
        """
        Initializes the ACS Workflow Guard.

        Args:
            policy_layer: Optional PolicyLayer for tool-level evaluation.
        """
        self.policy_layer = policy_layer or PolicyLayer()
        self.logger = logging.getLogger(__name__)
        self.fallback = RealtimeGuardrailFallback(self.policy_layer)

    def checkpoint_1_input_validation(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 1: Input Validation.
        Validates the raw input data.
        """
        if not isinstance(workflow_data, dict):
            return False, "Input validation failed: workflow data must be a dictionary."
        if not workflow_data:
            return False, "Input validation failed: workflow data is empty."

        required_fields = ["action", "tool"]
        for field in required_fields:
            if field not in workflow_data:
                return False, f"Input validation failed: missing '{field}' field."
            if not isinstance(workflow_data[field], str):
                return False, f"Input validation failed: '{field}' must be a string."

        return True, "Input validation passed."

    def checkpoint_2_intent_authorization(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 2: Intent Authorization.
        Verifies if the agent's intent is authorized.
        """
        action = workflow_data.get("action")
        allowed_intents = {"read", "write", "execute", "plan", "reflect", "delegate", "analyze"}

        if action == "unauthorized_action":
            return False, f"Intent authorization failed: action '{action}' is explicitly blacklisted."

        if action not in allowed_intents:
            return False, f"Intent authorization failed: action '{action}' is not in allowed intents list."

        return True, "Intent authorization passed."

    def checkpoint_3_tool_policy(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 3: Tool Policy.
        Checks if the tool complies with defined policies.
        """
        tool = workflow_data.get("tool")
        if tool == "forbidden_tool":
            return False, f"Tool policy failed: tool '{tool}' is forbidden."

        kwargs = workflow_data.get("kwargs", {})
        allow, explanation = self.policy_layer.evaluate(tool, **kwargs)
        if not allow:
            return False, f"Tool policy failed: {explanation}"

        return True, "Tool policy passed."

    def checkpoint_4_state_transition(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 4: State Transition.
        Ensures the proposed state transition is valid.
        """
        current_state = workflow_data.get("current_state", "idle")
        next_state = workflow_data.get("next_state")

        if not next_state:
            return False, "State transition failed: next_state is missing."

        allowed_transitions = {
            "idle": ["planning", "reflecting", "analyzing"],
            "planning": ["executing", "idle"],
            "executing": ["evaluating", "idle"],
            "evaluating": ["idle", "planning"],
            "reflecting": ["idle"],
            "analyzing": ["idle", "planning"],
            "error": ["idle"]
        }

        if current_state not in allowed_transitions:
             return False, f"State transition failed: unknown current_state '{current_state}'."

        if next_state not in allowed_transitions[current_state] and next_state != "error":
            return False, f"State transition failed: cannot transition from '{current_state}' to '{next_state}'."

        return True, "State transition passed."

    def checkpoint_5_output_sanitization(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 5: Output Sanitization.
        Sanitizes the final output.
        """
        output = workflow_data.get("output", "")
        output_str = str(output)

        if "secret_key" in output_str:
            return False, "Output sanitization failed: 'secret_key' keyword detected."

        for pattern in _SENSITIVE_PATTERNS:
            if pattern.search(output_str):
                return False, f"Output sanitization failed: sensitive pattern '{pattern.pattern}' detected."

        return True, "Output sanitization passed."

    def validate_workflow(self, workflow_data: Dict[str, Any]) -> bool:
        """
        Validates the workflow data through all 5 ACS checkpoints.
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

    def validate_with_fallback(self, workflow_data: Dict[str, Any], fallback_action: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates the workflow data through all 5 ACS checkpoints with a realtime fallback.
        """
        passed = self.validate_workflow(workflow_data)
        if passed:
            return True, workflow_data

        if fallback_action is not None:
            self.logger.info("ACS validation failed, triggering fallback action.")
            return False, fallback_action

        self.logger.warning("ACS validation failed and no fallback provided. Returning error state.")
        return False, {
            "action": "error",
            "tool": "none",
            "current_state": workflow_data.get("current_state", "unknown"),
            "next_state": "error",
            "output": "Action blocked by safety guardrails."
        }

    def intercept_action(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intercepts a state-changing action, validates it, and raises an exception if invalid.
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
