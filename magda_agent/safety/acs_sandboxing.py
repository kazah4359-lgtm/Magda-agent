"""ACS Guard Tool Sandboxing."""
from typing import Any, Callable, Dict, Set
from magda_agent.safety.taint_tracking_v2 import TaintTrackerV2, SandboxExecutionEnvironmentV2, PolicyViolationError

class ACSToolSandbox:
    """A sandbox for MCP tool executions dynamically with taint tracking."""
    def __init__(self) -> None:
        """Initialize the ACS tool sandbox."""
        self.tracker = TaintTrackerV2()
        self.sandbox = SandboxExecutionEnvironmentV2(self.tracker)
        self.restricted_tools: Set[str] = set()

    def restrict_tool(self, tool_name: str) -> None:
        """Mark a tool as restricted, meaning it cannot accept tainted data."""
        self.restricted_tools.add(tool_name)

    def execute_tool(self, tool_name: str, tool_func: Callable[..., Any], **kwargs: Any) -> Any:
        """Execute a tool within the sandbox dynamically."""
        is_restricted = tool_name in self.restricted_tools

        if is_restricted:
            if self.tracker.is_tainted(kwargs):
                origins = self.tracker.get_origins(kwargs)
                raise PolicyViolationError(f"Tainted input to restricted tool '{tool_name}' from origins: {origins}")

        try:
            return self.sandbox.execute(tool_func, **kwargs)
        except Exception as e:
            if isinstance(e, PolicyViolationError):
                raise
            raise RuntimeError(f"Tool execution failed: {str(e)}")
