"""MCP Kernel Taint Tracking Sandbox v5.

Inspired by MCPKernel taint tracking trend: Sandbox tool execution in containers to improve security.
Provides a sandbox execution environment for MCP tools with taint tracking, supporting both
synchronous and asynchronous executions, and timeout limits simulating container resource isolation.
"""
import asyncio
import inspect
from typing import Any, Callable, Dict, TypeVar

from magda_agent.safety.taint_tracking_v2 import (
    PolicyViolationError,
    SandboxExecutionEnvironmentV2,
    TaintTrackerV2,
)

T = TypeVar("T")

class ContainerIsolationError(Exception):
    """Raised when sandbox container limits (like timeouts) are exceeded."""
    pass


class MCPKernelSandboxV5:
    """
    Sandbox execution environment with taint tracking capabilities inspired by MCPKernel.
    Version 5 introduces async support and simulated container-level timeouts.
    """

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        """
        Initialize the MCPKernelSandboxV5.

        Args:
            timeout_seconds: The maximum time (in seconds) allowed for an async tool to execute
                             before simulating a container isolation timeout.
        """
        self.tracker = TaintTrackerV2()
        self.sandbox = SandboxExecutionEnvironmentV2(self.tracker)
        self.timeout_seconds = timeout_seconds

    def execute(self, tool_func: Callable[..., Any], inputs: Dict[str, Any], is_sensitive: bool = False) -> Any:
        """
        Executes a synchronous tool within the sandbox boundary.

        Args:
            tool_func: The synchronous function to execute.
            inputs: A dictionary of inputs to pass to the function.
            is_sensitive: If True, execution will fail if inputs contain tainted data.

        Returns:
            The result of the tool execution, appropriately tainted if inputs were tainted.

        Raises:
            PolicyViolationError: If is_sensitive is True and inputs contain tainted data.
            RuntimeError: If the tool execution fails.
            ValueError: If tool_func is an asynchronous coroutine function.
        """
        if inspect.iscoroutinefunction(tool_func):
            raise ValueError("execute() does not support coroutine functions. Use execute_async() instead.")

        self._check_taint(inputs, is_sensitive)

        try:
            return self.sandbox.execute(tool_func, **inputs)
        except PolicyViolationError:
            raise
        except Exception as e:
            raise RuntimeError(f"Sandbox execution failed: {str(e)}") from e

    async def execute_async(self, tool_func: Callable[..., Any], inputs: Dict[str, Any], is_sensitive: bool = False) -> Any:
        """
        Executes an asynchronous tool within the sandbox boundary with timeout limits.

        Args:
            tool_func: The asynchronous coroutine function to execute.
            inputs: A dictionary of inputs to pass to the function.
            is_sensitive: If True, execution will fail if inputs contain tainted data.

        Returns:
            The result of the async tool execution, appropriately tainted if inputs were tainted.

        Raises:
            PolicyViolationError: If is_sensitive is True and inputs contain tainted data.
            ContainerIsolationError: If the execution exceeds the sandbox timeout limit.
            RuntimeError: If the tool execution fails.
            ValueError: If tool_func is a regular synchronous function.
        """
        if not inspect.iscoroutinefunction(tool_func):
            raise ValueError("execute_async() requires a coroutine function. Use execute() instead.")

        self._check_taint(inputs, is_sensitive)

        # To propagate taint correctly, we need to gather origins manually here for async functions
        # since SandboxExecutionEnvironmentV2.execute is synchronous and cannot await.
        # So we replicate the tracking logic for async wrapper.

        origins = set()
        for val in inputs.values():
            origins.update(self.tracker.get_origins(val))

        try:
            # Enforce container timeout isolation
            result = await asyncio.wait_for(tool_func(**inputs), timeout=self.timeout_seconds)

            if origins:
                return self.tracker.taint_with_origins(result, origins)
            return result

        except asyncio.TimeoutError as e:
            raise ContainerIsolationError(f"Execution timed out after {self.timeout_seconds} seconds") from e
        except Exception as e:
            raise RuntimeError(f"Sandbox async execution failed: {str(e)}") from e

    def _check_taint(self, inputs: Dict[str, Any], is_sensitive: bool) -> None:
        """Check for taint violation if the tool is sensitive."""
        if is_sensitive and self.tracker.is_tainted(inputs):
            origins = self.tracker.get_origins(inputs)
            raise PolicyViolationError(f"Tainted input to sensitive tool call from origins: {origins}")
