"""MCP Policy Runtime Engine Isolation v1.

Inspired by MCP trends: Introduce an isolated runtime execution environment
for untrusted MCP tools using a taint policy engine.
"""

from typing import Any, Callable, Dict, Optional
from magda_agent.safety.mcp_taint_policy import MCPTaintPolicyEngine
from magda_agent.safety.mcp_kernel_sandbox import MCPKernelSandbox

class MCPIsolationEngine:
    """Provides an isolated runtime execution environment for untrusted tools."""

    def __init__(self, policy_engine: Optional[MCPTaintPolicyEngine] = None) -> None:
        """Initialize the isolation engine."""
        self.policy_engine = policy_engine or MCPTaintPolicyEngine()
        self.sandbox = MCPKernelSandbox()
        # Ensure they share the same taint tracker
        self.sandbox.tracker = self.policy_engine.tracker
        self.sandbox.sandbox.tracker = self.policy_engine.tracker

    def execute_untrusted_tool(
        self,
        tool_func: Callable[..., Any],
        inputs: Dict[str, Any],
    ) -> Any:
        """
        Execute an untrusted tool in an isolated environment.

        Args:
            tool_func: The function to execute.
            inputs: A dictionary of inputs to pass to the function.

        Returns:
            The result of the tool execution.

        Raises:
            PolicyViolationError: If the output contains tainted data.
            RuntimeError: If the tool execution fails.
        """
        # Evaluate inputs before execution
        self.policy_engine.evaluate_stream(inputs, is_sensitive=False)

        # Execute tool within sandbox
        result = self.sandbox.execute(tool_func, inputs, is_sensitive=False)

        # Evaluate output stream. For untrusted tools, we assume is_trusted=False.
        # This will raise PolicyViolationError if the output is tainted.
        self.policy_engine.evaluate_output_stream(result, is_trusted=False)

        return result
