import logging
from typing import Dict, Any, Callable, Tuple, Optional
from magda_agent.safety.acs import ACSWorkflowGuard

class GuardrailViolationError(Exception):
    """Exception raised when an action is blocked by a guardrail checkpoint."""
    def __init__(self, message: str, feedback: Dict[str, Any] = None):
        super().__init__(message)
        self.feedback = feedback or {}

class ACSGuardrailsV2:
    """
    Robust guardrail system with explicit validation checkpoints before executing tools and emitting outputs,
    based on Microsoft's Agent Control Specification (ACS).
    """
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.legacy_guard = ACSWorkflowGuard()

    def pre_tool_checkpoint(self, tool_name: str, kwargs: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates the tool and its arguments before execution.
        Returns: (is_passed, reason, feedback)
        """
        if not tool_name or not isinstance(tool_name, str):
            feedback = {"error": "Invalid tool name", "resolution": "Provide a valid string tool_name."}
            return False, "Pre-tool checkpoint failed: invalid tool name.", feedback

        workflow_data = {
            "action": "execute",
            "tool": tool_name,
            "kwargs": kwargs
        }

        # Checkpoint 1: Input Validation
        passed, reason = self.legacy_guard.checkpoint_1_input_validation(workflow_data)
        if not passed:
            return False, reason, {"error": "Input validation failed", "resolution": reason}

        # Checkpoint 3: Tool Policy
        passed, reason = self.legacy_guard.checkpoint_3_tool_policy(workflow_data)
        if not passed:
            return False, reason, {"error": "Tool policy failed", "resolution": reason}

        # Add custom pre-tool rules from V2 requirements
        if "malicious_arg" in kwargs:
             feedback = {"error": "Malicious argument detected", "resolution": "Remove 'malicious_arg' from arguments."}
             return False, "Pre-tool checkpoint failed: argument validation failed.", feedback

        return True, "Pre-tool checkpoint passed.", {}

    def pre_output_checkpoint(self, output: Any) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates the tool output before returning to the caller.
        Returns: (is_passed, reason, feedback)
        """
        workflow_data = {
            "output": output
        }

        # Checkpoint 5: Output Sanitization
        passed, reason = self.legacy_guard.checkpoint_5_output_sanitization(workflow_data)
        if not passed:
            return False, reason, {"error": "Output sanitization failed", "resolution": reason}

        return True, "Pre-output checkpoint passed.", {}

    def execute_with_guardrails(self, tool_name: str, kwargs: Dict[str, Any], tool_func: Callable[..., Any]) -> Any:
        """
        Wraps a tool execution with pre-tool and pre-output checkpoints.
        Raises GuardrailViolationError if any checkpoint fails.
        """
        # Pre-tool checkpoint
        passed, reason, feedback = self.pre_tool_checkpoint(tool_name, kwargs)
        if not passed:
            self.logger.warning(reason)
            raise GuardrailViolationError(reason, feedback)

        self.logger.debug(f"Executing tool {tool_name} with kwargs {kwargs}")

        # Execute tool
        try:
            output = tool_func(**kwargs)
        except Exception as e:
            self.logger.error(f"Tool {tool_name} raised exception: {e}")
            raise

        # Pre-output checkpoint
        passed, reason, feedback = self.pre_output_checkpoint(output)
        if not passed:
            self.logger.warning(reason)
            raise GuardrailViolationError(reason, feedback)

        return output
