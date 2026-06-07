"""
Agent Guard module.

Provides a runtime governance layer that sits between the agent's tool execution
and the external world. It intercepts and evaluates tool calls according to predefined security policies.
"""

import logging
from typing import Callable, Any

from magda_agent.safety.policy import PolicyLayer


class SecurityViolationError(Exception):
    """Exception raised when an action is blocked by the Agent Guard."""
    pass


class AgentGuard:
    """
    Runtime governance layer that intercepts tool calls and evaluates them
    against security policies before execution.
    """

    def __init__(self, policy_layer: PolicyLayer) -> None:
        """
        Initializes the Agent Guard.

        Args:
            policy_layer: The policy layer used to evaluate actions.
        """
        self.policy_layer = policy_layer
        self.logger = logging.getLogger(__name__)

    def execute_tool(self, tool_func: Callable, tool_name: str, **kwargs: Any) -> Any:
        """
        Intercepts and evaluates a tool call before executing it.

        Args:
            tool_func: The actual tool function to execute if permitted.
            tool_name: The name of the tool/action to evaluate.
            **kwargs: The arguments to pass to the tool.

        Returns:
            The result of the tool execution if permitted.

        Raises:
            SecurityViolationError: If the action is blocked by the policy.
        """
        allow, explanation = self.policy_layer.evaluate(tool_name, **kwargs)

        if not allow:
            self.logger.warning(
                f"AgentGuard: Tool execution blocked for '{tool_name}'. Reason: {explanation}"
            )
            raise SecurityViolationError(f"Action '{tool_name}' blocked: {explanation}")

        self.logger.info(f"AgentGuard: Tool execution permitted for '{tool_name}'.")
        return tool_func(**kwargs)
