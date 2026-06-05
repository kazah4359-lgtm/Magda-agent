import logging
from typing import Tuple, Dict, Any, List

class PolicyLayer:
    """
    Policy Layer for all tool/action execution.
    Evaluates every action with an external effect before execution.
    Logs an audit trail of all allowed and denied actions.
    """

    def __init__(self) -> None:
        """
        Initializes the Policy Layer.
        """
        self.audit_trail: List[Dict[str, Any]] = []

    def evaluate(self, tool_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """
        Evaluates a tool call against the safety policy.

        Args:
            tool_name (str): The name of the tool or action.
            **kwargs: The arguments passed to the tool.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the action is allowed,
                              and an LLM-friendly explanation string.
        """
        allow = True
        explanation = f"Action '{tool_name}' is allowed."

        # Example policy rules
        if tool_name == "system_execute_code":
            # Check for sensitive paths
            code = kwargs.get("code", "")
            if ".env" in code or "secrets" in code:
                allow = False
                explanation = "Action denied: 'system_execute_code' cannot access sensitive paths like '.env' or 'secrets'."
        elif tool_name == "send_message":
            # Check for spam or blocked recipients
            recipient = kwargs.get("recipient", "")
            if recipient == "blocked_user":
                allow = False
                explanation = "Action denied: Cannot send message to a blocked recipient."

        # Audit trail
        audit_entry = {
            "tool_name": tool_name,
            "kwargs": kwargs,
            "allowed": allow,
            "explanation": explanation
        }
        self.audit_trail.append(audit_entry)

        if allow:
            logging.info(f"PolicyLayer: ALLOW - {tool_name} with args {kwargs}")
        else:
            logging.warning(f"PolicyLayer: DENY - {tool_name} with args {kwargs}. Reason: {explanation}")

        return allow, explanation

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """
        Retrieves the audit trail of all evaluated actions.

        Returns:
            List[Dict[str, Any]]: The list of audit entries.
        """
        return self.audit_trail
