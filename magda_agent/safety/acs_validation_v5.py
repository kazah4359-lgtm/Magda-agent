import logging
from typing import Dict, Any, Tuple


class ACSValidationV5:
    """
    ACS Validation V5.
    Implements runtime tool validation checks to block potentially destructive actions
    before execution.
    """

    def __init__(self) -> None:
        """Initializes the ACSValidationV5 instance."""
        self.logger = logging.getLogger(__name__)

    def validate_tool_call(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates the tool call specified in the workflow_data.

        Args:
            workflow_data: A dictionary containing at least 'tool' and 'kwargs' keys.

        Returns:
            A tuple containing a boolean indicating if the tool is safe to execute,
            and a string explaining the reason.
        """
        tool = workflow_data.get("tool")
        kwargs = workflow_data.get("kwargs", {})

        if not tool:
            return False, "Validation failed: 'tool' field is missing or empty."

        if not isinstance(kwargs, dict):
            return False, "Validation failed: 'kwargs' must be a dictionary."

        # Block specific destructive tools entirely
        destructive_tools = {"rm", "format", "delete_database", "shutdown"}
        if tool in destructive_tools:
            return False, f"Validation failed: Tool '{tool}' is considered destructive and blocked."

        # Block potentially dangerous payloads for execution tools
        execution_tools = {"system_execute_code", "run_in_bash_session"}
        if tool in execution_tools:
            command = str(kwargs.get("command", "")).lower()
            code = str(kwargs.get("code", "")).lower()
            payload = command + " " + code

            # Simple heuristic for destructive payloads
            dangerous_patterns = ["rm -rf", "mkfs", "dd if=", "> /dev/sda", "drop table"]
            for pattern in dangerous_patterns:
                if pattern in payload:
                    return False, f"Validation failed: Tool '{tool}' contains dangerous pattern '{pattern}'."

        # Allow if no checks failed
        return True, "Validation passed: Tool is considered safe."
