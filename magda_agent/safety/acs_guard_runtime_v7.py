import logging
import re
from typing import Dict, Any, Tuple, Optional, Callable

from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.audit_trail import AuditTrail
from magda_agent.safety.taint import is_tainted
from magda_agent.safety.acs_sandboxing import ACSToolSandbox
from magda_agent.safety.acs_persistence import ACSPersistence
from magda_agent.safety.acs_state_transition_v5 import ACSStateTransitionV5

_SENSITIVE_PATTERNS = (
    re.compile(r"(?:api[_-]?key|access[_-]?token|bearer[_-]?token|auth[_-]?token|password|private[_-]?key|secret[_-]?key)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
    re.compile(r"\bAWS_SECRET_ACCESS_KEY\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\.env\b", re.IGNORECASE),
)


class SecurityViolationError(Exception):
    """Exception raised when an action or execution is blocked by the ACS Guard."""
    pass


class ACSRuntimeViolationError(SecurityViolationError):
    """Exception raised when a runtime execution or policy checkpoint fails."""
    pass


class ACSGuardRuntimeV7:
    """
    ACS (Agent Control Specification) Runtime Guard V7.
    Provides a robust runtime validation guardrail module that intercepts tool executions,
    evaluating them against 5 ACS validation checkpoints for policy adherence.
    """

    def __init__(
        self,
        policy_layer: Optional[PolicyLayer] = None,
        audit_trail: Optional[AuditTrail] = None,
        persistence: Optional[ACSPersistence] = None
    ) -> None:
        """
        Initializes the ACS Guard Runtime V7.

        Args:
            policy_layer: An optional policy layer for evaluating tool policies.
            audit_trail: An optional audit trail for logging checkpoint results.
            persistence: An optional persistence layer for recording checkpoint states.
        """
        self.logger = logging.getLogger(__name__)
        self.policy_layer = policy_layer or PolicyLayer()
        self.audit_trail = audit_trail or AuditTrail()
        self.persistence = persistence or ACSPersistence()
        self.tool_sandbox = ACSToolSandbox()
        self.state_transition_guard = ACSStateTransitionV5()

    def checkpoint_1_input_validation(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 1: Input Validation.
        Ensures the raw input data for the action is well-formed dictionary and free of taint.
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

        if is_tainted(workflow_data.get("action")):
            return False, "Input validation failed: action is tainted."

        if is_tainted(workflow_data.get("tool")):
            return False, "Input validation failed: tool name is tainted."

        kwargs = workflow_data.get("kwargs", {}) or {}
        if is_tainted(kwargs):
            return False, "Input validation failed: tainted data detected in tool inputs (kwargs)."

        return True, "Input validation passed."

    def checkpoint_2_intent_authorization(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 2: Intent Authorization.
        Verifies if the agent's intent is authorized and not blacklisted.
        """
        action = workflow_data.get("action")
        if is_tainted(action):
            return False, "Intent authorization failed: action is tainted."

        allowed_intents = {
            "read", "write", "execute", "plan", "reflect", "delegate", "analyze", "chat", "search"
        }

        if action == "unauthorized_action":
            return False, f"Intent authorization failed: action '{action}' is explicitly blacklisted."

        if action not in allowed_intents:
            return False, f"Intent authorization failed: action '{action}' is not in allowed intents list."

        return True, "Intent authorization passed."

    def checkpoint_3_tool_policy(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checkpoint 3: Tool Policy.
        Checks if the tool complies with defined runtime policies using PolicyLayer.
        """
        tool = workflow_data.get("tool")
        if is_tainted(tool):
            return False, "Tool policy failed: tool name is tainted."

        if tool == "forbidden_tool":
            return False, f"Tool policy failed: tool '{tool}' is forbidden."

        kwargs = workflow_data.get("kwargs", {}) or {}
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
        return self.state_transition_guard.checkpoint_4_state_transition(workflow_data)

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
            workflow_data: The workflow context dictionary.

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
            self.logger.debug(f"ACS Checkpoint {i} Passed: {reason}")

        return True

    def intercept_action(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intercepts an action, validates it through all 5 checkpoints,
        logs the results to AuditTrail and ACSPersistence, and raises SecurityViolationError on failure.

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

        kwargs = workflow_data.get("kwargs", {}) or {}

        for i, checkpoint in enumerate(checkpoints, 1):
            passed, reason = checkpoint(workflow_data)
            status = "passed" if passed else "failed"
            self.persistence.log_checkpoint(
                checkpoint_id=i,
                status=status,
                reason=reason,
                workflow_context=workflow_data
            )

            if not passed:
                self.logger.warning(f"ACS Checkpoint {i} Failed: {reason}")
                self.audit_trail.log_call(
                    tool_name=workflow_data.get("tool", "unknown"),
                    kwargs=kwargs,
                    why=f"ACS Checkpoint {i} Failed: {reason}",
                    result="blocked",
                    duration=0.0
                )
                raise ACSRuntimeViolationError(f"Action blocked by ACS checkpoint {i}: {reason}")
            self.logger.debug(f"ACS Checkpoint {i} Passed: {reason}")

        self.audit_trail.log_call(
            tool_name=workflow_data.get("tool", "unknown"),
            kwargs=kwargs,
            why="All 5 ACS checkpoints passed.",
            result="allowed",
            duration=0.0
        )

        return workflow_data

    def execute_with_guard(
        self,
        func: Callable[..., Any],
        workflow_data: Dict[str, Any],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Executes a tool function wrapped with pre-execution (checkpoints 1-4)
        and post-execution (checkpoint 5) ACS validation checks.

        Args:
            func: The tool function to execute.
            workflow_data: Workflow metadata dictionary describing action, tool, kwargs, state.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Any: The function execution output if all checkpoints pass.

        Raises:
            ACSRuntimeViolationError: If pre or post execution checkpoints fail.
        """
        tool_kwargs = workflow_data.get("kwargs", {}) or {}

        # Pre-execution checkpoints (1 to 4)
        pre_checkpoints = [
            (1, self.checkpoint_1_input_validation),
            (2, self.checkpoint_2_intent_authorization),
            (3, self.checkpoint_3_tool_policy),
            (4, self.checkpoint_4_state_transition)
        ]

        for i, checkpoint in pre_checkpoints:
            passed, reason = checkpoint(workflow_data)
            self.persistence.log_checkpoint(
                checkpoint_id=i,
                status="passed" if passed else "failed",
                reason=reason,
                workflow_context=workflow_data
            )
            if not passed:
                self.audit_trail.log_call(
                    tool_name=workflow_data.get("tool", "unknown"),
                    kwargs=tool_kwargs,
                    why=f"ACS Checkpoint {i} Failed: {reason}",
                    result="blocked",
                    duration=0.0
                )
                raise ACSRuntimeViolationError(f"Pre-execution blocked by ACS checkpoint {i}: {reason}")

        # Execute tool function
        output = func(*args, **kwargs)

        # Post-execution checkpoint (5: Output Sanitization)
        post_data = dict(workflow_data)
        post_data["output"] = output

        passed_5, reason_5 = self.checkpoint_5_output_sanitization(post_data)
        self.persistence.log_checkpoint(
            checkpoint_id=5,
            status="passed" if passed_5 else "failed",
            reason=reason_5,
            workflow_context=post_data
        )

        if not passed_5:
            self.audit_trail.log_call(
                tool_name=workflow_data.get("tool", "unknown"),
                kwargs=tool_kwargs,
                why=f"ACS Checkpoint 5 Failed: {reason_5}",
                result="blocked",
                duration=0.0
            )
            raise ACSRuntimeViolationError(f"Post-execution blocked by ACS checkpoint 5: {reason_5}")

        self.audit_trail.log_call(
            tool_name=workflow_data.get("tool", "unknown"),
            kwargs=tool_kwargs,
            why="All 5 ACS checkpoints passed during guarded execution.",
            result="allowed",
            duration=0.0
        )

        return output


ACSGuard = ACSGuardRuntimeV7
ACSGuardV7 = ACSGuardRuntimeV7
