"""MCP Kernel Taint Tracking Sandbox v1.

Provides a sandbox execution environment for MCP tools with taint tracking.
"""
from typing import Any, Callable, Dict

from magda_agent.safety.taint_tracking_v2 import (
    TaintTrackerV2,
    SandboxExecutionEnvironmentV2,
    PolicyViolationError,
)

class MCPKernelSandbox:
    """Sandbox execution environment with taint tracking capabilities inspired by MCPKernel."""

    def __init__(self) -> None:
        """Initialize the MCPKernelSandbox."""
        self.tracker = TaintTrackerV2()
        self.sandbox = SandboxExecutionEnvironmentV2(self.tracker)

    def execute(self, tool_func: Callable[..., Any], inputs: Dict[str, Any], is_sensitive: bool = False) -> Any:
        """
        Executes a tool within the sandbox boundary.

        Args:
            tool_func: The function to execute.
            inputs: A dictionary of inputs to pass to the function.
            is_sensitive: If True, execution will fail if inputs contain tainted data.

        Returns:
            The result of the tool execution, appropriately tainted if inputs were tainted.

        Raises:
            PolicyViolationError: If is_sensitive is True and inputs contain tainted data.
            RuntimeError: If the tool execution fails.
        """
        if is_sensitive and self.tracker.is_tainted(inputs):
            origins = self.tracker.get_origins(inputs)
            raise PolicyViolationError(f"Tainted input to sensitive tool call from origins: {origins}")

        try:
            return self.sandbox.execute(tool_func, **inputs)
        except PolicyViolationError:
            raise
        except Exception as e:
            raise RuntimeError(f"Sandbox execution failed: {str(e)}")
