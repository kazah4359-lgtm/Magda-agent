"""
Runtime Governance Layer module.

Inspired by ACS Specification trend: this module provides a runtime governance
layer that evaluates every tool call against policy rules before execution.
It relies on PolicyLayer for the actual evaluation.
"""

import inspect
import logging
from typing import Callable, Any
from functools import wraps

from magda_agent.safety.policy import PolicyLayer


class GovernanceViolationError(Exception):
    """Exception raised when a tool execution violates a governance policy."""
    pass


class RuntimeGovernanceLayer:
    """
    Evaluates tool calls against predefined policies prior to execution.
    """

    def __init__(self, policy_layer: PolicyLayer) -> None:
        """
        Initializes the RuntimeGovernanceLayer.

        Args:
            policy_layer: The policy layer used for evaluating tool calls.
        """
        self.policy_layer = policy_layer
        self.logger = logging.getLogger(__name__)

    def execute_tool(self, tool_func: Callable, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Intercepts and evaluates a tool call before executing it.

        Args:
            tool_func: The actual tool function to execute if permitted.
            tool_name: The name of the tool being executed.
            **kwargs: Arguments to pass to the tool.

        Returns:
            The result of the tool execution if permitted.

        Raises:
            GovernanceViolationError: If the execution is blocked by the policy layer.
        """
        allow, explanation = self.policy_layer.evaluate(tool_name, *args, **kwargs)

        if not allow:
            self.logger.warning(f"RuntimeGovernanceLayer: Execution blocked for '{tool_name}'. Reason: {explanation}")
            raise GovernanceViolationError(f"Execution of '{tool_name}' blocked: {explanation}")

        self.logger.info(f"RuntimeGovernanceLayer: Execution permitted for '{tool_name}'.")

        # Check if the tool function is asynchronous
        if inspect.iscoroutinefunction(tool_func):
            async def async_wrapper() -> Any:
                return await tool_func(*args, **kwargs)
            return async_wrapper()

        # Handle synchronous execution
        result = tool_func(*args, **kwargs)

        # Allow returning coroutines from synchronous wrapper functions
        if inspect.isawaitable(result):
            return result

        return result

    def governance_guard(self, tool_name: str) -> Callable:
        """
        A decorator to wrap a tool function with the RuntimeGovernanceLayer.

        Args:
            tool_name: The name of the tool for policy evaluation.

        Returns:
            The decorated tool function.
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return self.execute_tool(func, tool_name, *args, **kwargs)

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await self.execute_tool(func, tool_name, *args, **kwargs)

            return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

        return decorator
