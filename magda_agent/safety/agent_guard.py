"""
Agent Guard module.

Provides a runtime governance layer that sits between the agent's tool execution
and the external world. It intercepts and evaluates tool calls according to predefined security policies.
"""

import inspect
import logging
from typing import Callable, Any, Coroutine, Union
from functools import wraps

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
        Supports both synchronous and asynchronous tool functions.

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

        # Check if the tool function is a coroutine function
        if inspect.iscoroutinefunction(tool_func):
            async def async_wrapper() -> Any:
                return await tool_func(**kwargs)
            return async_wrapper()

        # Handle synchronous functions normally
        result = tool_func(**kwargs)

        # If the synchronous function returned a coroutine, return it directly
        # (similar to execute_skill in registry.py)
        if inspect.isawaitable(result):
            return result

        return result

    def guard_tool(self, tool_name: str) -> Callable:
        """
        A decorator to protect a tool function with the Agent Guard.

        Args:
            tool_name: The name of the tool/action to evaluate.

        Returns:
            The decorated function.
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                # Merge args into kwargs for policy evaluation (assuming named args or ignoring positional)
                # In a real implementation, you'd map args to param names using inspect.signature
                # For this simple version, we pass only kwargs to the policy evaluator
                return self.execute_tool(func, tool_name, **kwargs)

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                return await self.execute_tool(func, tool_name, **kwargs)

            return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

        return decorator
