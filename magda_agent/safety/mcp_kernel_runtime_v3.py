"""
MCP Kernel Runtime V3.

Inspired by MCPKernel taint tracking trend: this module provides a runtime proxy
that enforces taint policies for tool calls.
"""

import inspect
from functools import wraps
from typing import Any, Callable, Dict, Optional, Coroutine

from magda_agent.safety.mcp_taint_policy import MCPTaintPolicyEngine
from magda_agent.safety.taint_tracking_v2 import PolicyViolationError


class MCPKernelRuntimeV3:
    """
    Acts as a runtime proxy enforcing taint policy for tool calls.
    """

    def __init__(self, policy_engine: Optional[MCPTaintPolicyEngine] = None) -> None:
        """
        Initializes the MCPKernelRuntimeV3.

        Args:
            policy_engine: The policy engine used for evaluating data streams.
                If not provided, a new one is created.
        """
        self.policy_engine = policy_engine or MCPTaintPolicyEngine()

    def execute_tool(
        self,
        tool_func: Callable[..., Any],
        inputs: Dict[str, Any],
        is_sensitive: bool = False,
        is_trusted: bool = False,
    ) -> Any:
        """
        Evaluates the input stream, executes the tool, and evaluates the output stream.

        Args:
            tool_func: The actual tool function to execute if permitted.
            inputs: A dictionary of inputs to pass to the tool.
            is_sensitive: If True, the tool is considered sensitive and execution will
                fail if inputs contain any tainted data.
            is_trusted: If True, the output is considered trusted and execution will
                succeed even if it contains tainted data.

        Returns:
            The result of the tool execution if permitted.

        Raises:
            PolicyViolationError: If input/output policies are violated.
        """
        # 1. Evaluate Input Stream
        self.policy_engine.evaluate_stream(inputs, is_sensitive=is_sensitive)

        # 2. Execute Tool
        # We handle coroutine vs sync inside runtime_proxy, but we also support direct calls
        if inspect.iscoroutinefunction(tool_func):
            async def async_wrapper() -> Any:
                result = await tool_func(**inputs)
                self.policy_engine.evaluate_output_stream(result, is_trusted=is_trusted)
                return result
            return async_wrapper()

        result = tool_func(**inputs)

        # Handle synchronous execution returning an awaitable (e.g. from wrapper)
        if inspect.isawaitable(result):
            async def async_result_wrapper() -> Any:
                awaited_result = await result
                self.policy_engine.evaluate_output_stream(awaited_result, is_trusted=is_trusted)
                return awaited_result
            return async_result_wrapper()

        # 3. Evaluate Output Stream
        self.policy_engine.evaluate_output_stream(result, is_trusted=is_trusted)

        return result

    def runtime_proxy(self, is_sensitive: bool = False, is_trusted: bool = False) -> Callable:
        """
        A decorator to wrap a tool function with the MCPKernelRuntimeV3.

        Args:
            is_sensitive: If True, the tool is considered sensitive.
            is_trusted: If True, the output is considered trusted.

        Returns:
            The decorated tool function.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # To be generic, combine args and kwargs into an inputs dictionary
                # For simplicity, we just package everything into a dict.
                # Real MCP tools often take named parameters (kwargs).
                # We'll map args to positional indices if necessary.

                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                inputs = dict(bound_args.arguments)

                # Check inputs
                self.policy_engine.evaluate_stream(inputs, is_sensitive=is_sensitive)

                result = func(*args, **kwargs)

                # Check if the synchronous wrapper returned a coroutine
                if inspect.isawaitable(result):
                     async def async_eval() -> Any:
                         awaited_result = await result
                         self.policy_engine.evaluate_output_stream(awaited_result, is_trusted=is_trusted)
                         return awaited_result
                     return async_eval()

                # Check output
                self.policy_engine.evaluate_output_stream(result, is_trusted=is_trusted)

                return result

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                inputs = dict(bound_args.arguments)

                # Check inputs
                self.policy_engine.evaluate_stream(inputs, is_sensitive=is_sensitive)

                result = await func(*args, **kwargs)

                # Check output
                self.policy_engine.evaluate_output_stream(result, is_trusted=is_trusted)

                return result

            return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

        return decorator
