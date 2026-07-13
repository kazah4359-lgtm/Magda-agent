import logging
import re
from typing import Dict, Any, Tuple, Optional
from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.audit_trail import AuditTrail
from magda_agent.safety.taint import is_tainted
from magda_agent.safety.acs_sandboxing import ACSToolSandbox
from magda_agent.safety.acs_sandboxing import ACSToolSandbox

_SENSITIVE_PATTERNS = (
    re.compile(r"api[_-]?key|token|password|private[_-]?key|secret[_-]?key", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
    re.compile(r"\.env", re.IGNORECASE),
    re.compile(r"secrets?", re.IGNORECASE),
)

class SecurityViolationError(Exception):
    """Exception raised when an action is blocked by the ACS Guard V6."""
    pass

class ACSGuardV6:
    """
    ACS (Agent Control Specification) Guard V6.
    Advanced runtime safety policy to intercept and validate tool execution
    across all skills using 5 validation checkpoints.
    """

    def __init__(self, policy_layer: Optional[PolicyLayer] = None, audit_trail: Optional[AuditTrail] = None) -> None:
        """
        Initializes the ACS Guard V6.

        Args:
            policy_layer: An optional policy layer for evaluating tool policies.
            audit_trail: An optional audit trail for logging checkpoint results.
        """
        self.logger = logging.getLogger(__name__)
        self.policy_layer = policy_layer or PolicyLayer()
        self.audit_trail = audit_trail or AuditTrail()
        self.tool_sandbox = ACSToolSandbox()

    def checkpoint_1_input_validation(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 1: Input Validation.
        Ensures the raw input data for the action is well-formed.
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

        if is_tainted(workflow_data.get("kwargs")):
            return False, "Input validation failed: tainted data detected in tool inputs (kwargs)."

        return True, "Input validation passed."

    def checkpoint_2_intent_authorization(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 2: Intent Authorization.
        Verifies if the agent's intent is authorized.
        """
        action = workflow_data.get("action")
        if is_tainted(action):
            return False, "Intent authorization failed: action is tainted."

        # Standard allowed intents in ACS-like architectures
        allowed_intents = {
            "read", "write", "execute", "plan", "reflect", "delegate", "analyze", "chat"
        }

        if action == "unauthorized_action":
            return False, f"Intent authorization failed: action '{action}' is explicitly blacklisted."

        if action not in allowed_intents:
            return False, f"Intent authorization failed: action '{action}' is not in allowed intents list."

        return True, "Intent authorization passed."

    def checkpoint_3_tool_policy(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 3: Tool Policy.
        Checks if the tool complies with defined policies using the PolicyLayer.
        """
        tool = workflow_data.get("tool")
        if is_tainted(tool):
            return False, "Tool policy failed: tool name is tainted."

        if tool == "forbidden_tool":
            return False, f"Tool policy failed: tool '{tool}' is forbidden."

        kwargs = workflow_data.get("kwargs", {})
        # Note: checkpoint_1 already checks for tainted kwargs,
        # but we re-check here for completeness if called in isolation.
        if is_tainted(kwargs):
            return False, "Tool policy failed: tainted data detected in tool inputs (kwargs)."

        allow, explanation = self.policy_layer.evaluate(tool, **kwargs)
        if not allow:
            return False, f"Tool policy failed: {explanation}"

        return True, "Tool policy passed."

    def checkpoint_4_state_transition(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 4: State Transition.
        Ensures the proposed state transition is valid within the cognitive architecture.
        """
        current_state = workflow_data.get("current_state", "idle")
        next_state = workflow_data.get("next_state")

        if not next_state:
            return True, "State transition passed: next_state not provided."

        allowed_transitions = {
            "idle": ["planning", "reflecting", "analyzing", "executing"],
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
        Sanitizes the final output to prevent leakage of sensitive data and checks for taint.
        """
        output = workflow_data.get("output")
        if output is None:
            return True, "Output sanitization passed: no output to sanitize."

        if is_tainted(output):
            return False, "Output sanitization failed: tainted data detected in output."

        output_str = str(output)

        for pattern in _SENSITIVE_PATTERNS:
            if pattern.search(output_str):
                return False, f"Output sanitization failed: sensitive pattern '{pattern.pattern}' detected."

        return True, "Output sanitization passed."

    def validate_action(self, workflow_data: Dict[str, Any]) -> bool:
        """
        Validates the workflow action through all 5 ACS checkpoints.

        Args:
            workflow_data: The workflow context data.

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
                self.logger.warning(f"ACS Checkpoint {i} Failed: {reason}")
                return False
            self.logger.info(f"ACS Checkpoint {i} Passed: {reason}")

        return True

    def intercept_action(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intercepts an action, validates it through all 5 checkpoints,
        logs the results to the AuditTrail, and raises SecurityViolationError on failure.

        Args:
            workflow_data: The workflow context data.

        Returns:
            Dict[str, Any]: The original workflow data if all checkpoints pass.

        Raises:
            SecurityViolationError: If any checkpoint fails.
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
                self.audit_trail.log_call(
                    tool_name=workflow_data.get("tool", "unknown"),
                    kwargs=workflow_data.get("kwargs", {}),
                    why=f"ACS Checkpoint {i} Failed: {reason}",
                    result="blocked",
                    duration=0.0
                )
                raise SecurityViolationError(f"Action blocked by ACS checkpoint {i}: {reason}")
            self.logger.debug(f"ACS Checkpoint {i} Passed: {reason}")

        self.audit_trail.log_call(
            tool_name=workflow_data.get("tool", "unknown"),
            kwargs=workflow_data.get("kwargs", {}),
            why="All 5 ACS checkpoints passed.",
            result="allowed",
            duration=0.0
        )

        return workflow_data
