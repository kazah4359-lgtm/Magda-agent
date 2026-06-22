"""MCPKernel Taint Tracking Sandbox.

Provides taint tracking and sandboxing for tool execution.
"""
from typing import Any, Callable, Dict, List


class PolicyViolationError(Exception):
    """Raised when tainted data violates policy."""
    pass


class TaintedData:
    """Base class for tainted data wrappers."""
    def __init__(self, value: Any):
        self.value = value


class TaintedString(TaintedData, str):
    """A string subclass that marks data as tainted."""
    def __new__(cls, value: str):
        obj = str.__new__(cls, value)
        return obj


def mark_tainted(s: Any) -> Any:
    """Marks a string, list, or dict as tainted recursively."""
    if isinstance(s, str) and not isinstance(s, TaintedString):
        return TaintedString(s)
    elif isinstance(s, list):
        return [mark_tainted(item) for item in s]
    elif isinstance(s, dict):
        return {mark_tainted(k): mark_tainted(v) for k, v in s.items()}
    return s


def is_tainted(s: Any) -> bool:
    """Checks if an object or any of its nested structures is tainted."""
    if isinstance(s, TaintedData):
        return True
    if isinstance(s, list):
        return any(is_tainted(item) for item in s)
    if isinstance(s, dict):
        return any(is_tainted(k) or is_tainted(v) for k, v in s.items())
    return False


def sanitize(s: Any) -> Any:
    """Sanitizes tainted data, returning regular primitives recursively."""
    if isinstance(s, TaintedData):
        return sanitize(s.value)
    elif isinstance(s, list):
        return [sanitize(item) for item in s]
    elif isinstance(s, dict):
        return {sanitize(k): sanitize(v) for k, v in s.items()}
    return s


class TaintTracker:
    """Tracks tainted objects."""
    def __init__(self) -> None:
        """Initialize the TaintTracker."""
        pass

    def taint(self, obj: Any) -> Any:
        """Mark an object as tainted recursively."""
        return mark_tainted(obj)

    def is_tainted(self, obj: Any) -> bool:
        """Check if an object or its contents are tainted."""
        return is_tainted(obj)

    def sanitize(self, obj: Any) -> Any:
        """Sanitize an object recursively."""
        return sanitize(obj)

    def clear(self) -> None:
        """Clear tracked objects. (No-op in this implementation)."""
        pass


class SandboxExecutionEnvironment:
    """A sandbox for executing tools with taint propagation."""
    def __init__(self, tracker: TaintTracker) -> None:
        """Initialize the sandbox execution environment."""
        self.tracker = tracker

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function in the sandbox.
        Automatically propagates taint from inputs to output.
        """
        any_input_tainted = any(self.tracker.is_tainted(arg) for arg in args) or \
                           any(self.tracker.is_tainted(val) for val in kwargs.values())

        result = func(*args, **kwargs)

        if any_input_tainted:
            return self.tracker.taint(result)
        return result


class MCPKernel:
    """Kernel for executing tools safely."""
    def __init__(self) -> None:
        """Initialize the MCPKernel."""
        self.tracker = TaintTracker()
        self.sandbox = SandboxExecutionEnvironment(self.tracker)

    def execute_tool(self, tool_func: Callable[..., Any], inputs: Dict[str, Any], is_sensitive: bool = False) -> Any:
        """Executes a tool within the kernel. If is_sensitive is True, tainted inputs will fail."""
        if is_sensitive:
            if self.tracker.is_tainted(inputs):
                raise PolicyViolationError(f"Tainted input to sensitive tool call: {inputs}")

        try:
            result = self.sandbox.execute(tool_func, **inputs)
            return result
        except Exception as e:
            if isinstance(e, PolicyViolationError):
                raise
            raise RuntimeError(f"Tool execution failed: {str(e)}")
