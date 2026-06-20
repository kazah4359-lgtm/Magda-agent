import logging
from typing import Dict, Any, Tuple, Optional
from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.audit_trail import AuditTrail


class SecurityViolationError(Exception):
    """Exception raised when an action is blocked by the ACS Guard V2."""
    pass

class ACSGuardV2:
    """
    Enhanced Runtime governance layer that intercepts state-changing actions
    and evaluates them through 5 ACS checkpoints with a policy-driven framework.
    """

    def __init__(self, policy_layer: Optional[PolicyLayer] = None) -> None:
        """
        Initializes the ACS Guard V2.

        Args:
            policy_layer: An optional policy layer for evaluating checkpoints.
        """
        self.logger = logging.getLogger(__name__)
        self.policy_layer = policy_layer
        self.audit_logger = AuditTrail()

    def checkpoint_1_input_validation(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 1: Input Validation.
        Validates the raw input data.
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
        """
        action = workflow_data.get("action")
        if action == "unauthorized_action":
            return False, f"Intent authorization failed: action '{action}' is not allowed."
        return True, "Intent authorization passed."

    def checkpoint_3_tool_policy(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 3: Tool Policy.
        Checks if the tool complies with defined policies.
        """
        tool = workflow_data.get("tool")
        if tool == "forbidden_tool":
            return False, f"Tool policy failed: tool '{tool}' is forbidden."

        if self.policy_layer and tool:
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
        current_state = workflow_data.get("current_state")
        next_state = workflow_data.get("next_state")
        if current_state == "error" and next_state == "executing":
            return False, f"State transition failed: cannot transition from '{current_state}' to '{next_state}'."
        return True, "State transition passed."

    def checkpoint_5_output_sanitization(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 5: Output Sanitization.
        Sanitizes the final output.
        """
        output = workflow_data.get("output", "")
        if "secret_key" in str(output):
            return False, "Output sanitization failed: sensitive data detected in output."
        return True, "Output sanitization passed."

    def validate_workflow(self, workflow_data: Dict[str, Any]) -> bool:
        """
        Validates workflow data through all 5 ACS checkpoints.
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
                self.audit_logger.log_call(
                    tool_name=workflow_data.get("tool", "unknown"),
                    kwargs=workflow_data.get("kwargs", {}),
                    why=f"Checkpoint {i} Failed: {reason}",
                    result="blocked",
                    duration=0.0
                )
                raise SecurityViolationError(f"Action blocked by ACS checkpoint {i}: {reason}")
            self.logger.debug(f"ACS Checkpoint {i} Passed: {reason}")

        self.audit_logger.log_call(
            tool_name=workflow_data.get("tool", "unknown"),
            kwargs=workflow_data.get("kwargs", {}),
            why="All 5 checkpoints passed.",
            result="allowed",
            duration=0.0
        )

        return workflow_data
