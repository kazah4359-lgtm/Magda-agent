import logging
from typing import Any, Tuple, Optional

from magda_agent.safety.policy import PolicyLayer


class ACSMemoryPolicy(PolicyLayer):
    """
    ACS Memory Policy Layer.
    Restricts sub-agent access to user-specific memory stores based on the user_id context.
    """

    def __init__(self, expected_user_id: Optional[str] = None) -> None:
        super().__init__()
        self.expected_user_id = expected_user_id

    def evaluate(self, tool_name: str, **kwargs: Any) -> Tuple[bool, str]:
        """
        Evaluates memory-related tool calls to ensure they contain a valid user_id context
        that matches the expected user_id.
        Delegates non-memory-related tool calls to the parent PolicyLayer.

        Args:
            tool_name: The name of the tool being executed.
            **kwargs: The arguments passed to the tool.

        Returns:
            A tuple of (allow, explanation).
        """
        if tool_name in ("read_memory", "write_memory"):
            user_id = kwargs.get("user_id")

            allow = True
            explanation = "Action allowed."

            if not user_id:
                allow = False
                explanation = f"Action denied: unauthorized memory access - missing user_id for '{tool_name}'."
            elif self.expected_user_id is not None and user_id != self.expected_user_id:
                allow = False
                explanation = f"Action denied: unauthorized memory access - user_id mismatch for '{tool_name}'."

            if not allow:
                self.audit_logger.log_call(
                    tool_name=tool_name,
                    kwargs=kwargs,
                    why=kwargs.get("why", "No reason provided"),
                    result={"allowed": allow, "explanation": explanation},
                    duration=0.0,
                )

                logging.warning(f"ACSMemoryPolicy: DENY - {tool_name} with args {kwargs}. Reason: {explanation}")
                return allow, explanation

        return super().evaluate(tool_name, **kwargs)
