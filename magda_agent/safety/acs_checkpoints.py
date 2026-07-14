import logging
import re
from enum import Enum
from typing import Dict, Any, Tuple, Optional, List, Callable
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

class CheckpointStage(Enum):
    """Stages of the Agent Control Specification validation pipeline."""
    INPUT = "input"
    EXECUTION = "execution"
    OUTPUT = "output"

class Checkpoint:
    """Represents a discrete validation checkpoint within the ACS pipeline."""

    def __init__(self, name: str, stage: CheckpointStage, validate_func: Callable[[Dict[str, Any]], Tuple[bool, str]]) -> None:
        """
        Initializes a Checkpoint instance.

        Args:
            name: Description/name of the checkpoint.
            stage: CheckpointStage enum value.
            validate_func: The validation function callable.
        """
        self.name = name
        self.stage = stage
        self.validate_func = validate_func

    def run(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Runs the checkpoint validation function.

        Args:
            action_data: The dictionary containing action context and payload.

        Returns:
            A tuple of (is_passed, message).
        """
        return self.validate_func(action_data)

class ACSCheckpoints:
    """
    Implements 5 ACS validation checkpoints for agentic workflows.
    Refactored to execute checkpoints through a formal pipeline of discrete stages.
    """

    def __init__(self, policy_layer: Optional[PolicyLayer] = None, audit_trail: Optional[AuditTrail] = None) -> None:
        """
        Initializes the ACSCheckpoints pipeline.

        Args:
            policy_layer: Optional PolicyLayer for tool evaluation.
            audit_trail: Optional AuditTrail for recording evaluation results.
        """
        self.logger = logging.getLogger(__name__)
        self.policy_layer = policy_layer or PolicyLayer()
        self.audit_trail = audit_trail or AuditTrail()

        # Build the pipeline of formal checkpoints
        self.pipeline: List[Checkpoint] = [
            Checkpoint(
                name="Input Validation",
                stage=CheckpointStage.INPUT,
                validate_func=self.checkpoint_1_input_validation
            ),
            Checkpoint(
                name="Intent Authorization",
                stage=CheckpointStage.INPUT,
                validate_func=self.checkpoint_2_intent_authorization
            ),
            Checkpoint(
                name="Tool Policy",
                stage=CheckpointStage.EXECUTION,
                validate_func=self.checkpoint_3_tool_policy
            ),
            Checkpoint(
                name="State Transition",
                stage=CheckpointStage.EXECUTION,
                validate_func=self.checkpoint_4_state_transition
            ),
            Checkpoint(
                name="Output Sanitization",
                stage=CheckpointStage.OUTPUT,
                validate_func=self.checkpoint_5_output_sanitization
            )
        ]

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

    def _run_stage(self, stage: CheckpointStage, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Executes all checkpoints belonging to a specific CheckpointStage.

        Args:
            stage: The CheckpointStage to execute.
            action_data: The dictionary containing action context and payload.

        Returns:
            A tuple of (is_passed, message).
        """
        for checkpoint in self.pipeline:
            if checkpoint.stage == stage:
                ok, reason = checkpoint.run(action_data)
                if not ok:
                    return False, reason
        return True, f"{stage.name} validation stage passed."

    def validate_pre_execution(self, action_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Runs INPUT and EXECUTION checkpoint stages and logs to audit trail on failure.

        Args:
            action_data: The dictionary containing action context and payload.

        Returns:
            A tuple of (is_passed, message).
        """
        for stage in [CheckpointStage.INPUT, CheckpointStage.EXECUTION]:
            ok, reason = self._run_stage(stage, action_data)
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
        """
        Runs OUTPUT checkpoint stage and logs to audit trail on failure.

        Args:
            action_data: The dictionary containing action context and payload.

        Returns:
            A tuple of (is_passed, message).
        """
        ok, reason = self._run_stage(CheckpointStage.OUTPUT, action_data)
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
        """
        Runs all 5 checkpoints across all stages and returns True if all pass.

        Args:
            action_data: The dictionary containing action context and payload.

        Returns:
            True if all checkpoints pass, False otherwise.
        """
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

        Args:
            action_data: The dictionary containing action context and payload.

        Returns:
            The input action_data if all checkpoints pass.

        Raises:
            SecurityViolationError: If any validation checkpoint fails.
        """
        if not self.validate_action(action_data):
            for checkpoint in self.pipeline:
                passed, reason = checkpoint.run(action_data)
                if not passed:
                     raise SecurityViolationError(reason)
        return action_data
