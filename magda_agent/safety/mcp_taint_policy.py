"""MCP Kernel Taint Tracking Policy Engine.

Inspired by MCPKernel taint tracking + sandboxed execution.
Provides a dynamic taint tracking policy engine to restrict untrusted tool outputs.
"""

from typing import Any, Dict

from magda_agent.safety.taint_tracking_v2 import PolicyViolationError, TaintTrackerV2


class MCPTaintPolicyEngine:
    """Evaluates data streams against taint policies before tool execution."""

    def __init__(self, tracker: TaintTrackerV2 | None = None) -> None:
        """
        Initialize the MCPTaintPolicyEngine.

        Args:
            tracker: An optional TaintTrackerV2 instance. If not provided, a new one is created.
        """
        self.tracker = tracker or TaintTrackerV2()

    def evaluate_stream(self, inputs: Dict[str, Any], is_sensitive: bool = False) -> None:
        """
        Evaluates a stream of inputs against the taint policy.

        Args:
            inputs: A dictionary of inputs to be passed to a tool.
            is_sensitive: If True, the tool is considered sensitive and execution will
                fail if inputs contain any tainted data.

        Raises:
            PolicyViolationError: If is_sensitive is True and inputs contain tainted data.
        """
        if is_sensitive and self.tracker.is_tainted(inputs):
            origins = self.tracker.get_origins(inputs)
            raise PolicyViolationError(
                f"Tainted input to sensitive tool call from origins: {origins}"
            )
