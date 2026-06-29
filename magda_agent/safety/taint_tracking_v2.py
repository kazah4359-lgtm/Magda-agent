"""MCPKernel Taint Tracking Sandbox V2.

Provides taint tracking with origin propagation and sandboxing for tool execution.
"""
from typing import Any, Callable, Dict, List, Set, Union


class PolicyViolationError(Exception):
    """Raised when tainted data violates policy."""
    pass


class TaintedData:
    """Base class for tainted data wrappers that track origins."""
    def __init__(self, value: Any, origins: Set[str] = None):
        self.value = value
        self.origins = origins if origins is not None else set()


class TaintedString(TaintedData, str):
    """A string subclass that marks data as tainted and tracks its origins."""
    def __new__(cls, value: str, origins: Set[str] = None):
        obj = str.__new__(cls, value)
        return obj

    def __init__(self, value: str, origins: Set[str] = None):
        TaintedData.__init__(self, value, origins)


def mark_tainted(s: Any, origin: str) -> Any:
    """Marks a string, list, or dict as tainted recursively with the given origin."""
    if isinstance(s, str) and not isinstance(s, TaintedString):
        return TaintedString(s, {origin})
    elif isinstance(s, TaintedString):
        s.origins.add(origin)
        return s
    elif isinstance(s, list):
        return [mark_tainted(item, origin) for item in s]
    elif isinstance(s, dict):
        return {mark_tainted(k, origin): mark_tainted(v, origin) for k, v in s.items()}
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


def get_origins(s: Any) -> Set[str]:
    """Retrieves all origins from a tainted object recursively."""
    origins = set()
    if isinstance(s, TaintedData):
        origins.update(s.origins)
    if isinstance(s, list):
        for item in s:
            origins.update(get_origins(item))
    if isinstance(s, dict):
        for k, v in s.items():
            origins.update(get_origins(k))
            origins.update(get_origins(v))
    return origins


def sanitize(s: Any) -> Any:
    """Sanitizes tainted data, returning regular primitives recursively."""
    if isinstance(s, TaintedString):
        return str(s.value)
    elif isinstance(s, TaintedData):
        return sanitize(s.value)
    elif isinstance(s, list):
        return [sanitize(item) for item in s]
    elif isinstance(s, dict):
        return {sanitize(k): sanitize(v) for k, v in s.items()}
    return s


class TaintTrackerV2:
    """Tracks tainted objects with origin tracing."""
    def __init__(self) -> None:
        """Initialize the TaintTrackerV2."""
        pass

    def taint(self, obj: Any, origin: str) -> Any:
        """Mark an object as tainted recursively with an origin."""
        return mark_tainted(obj, origin)

    def taint_with_origins(self, obj: Any, origins: Set[str]) -> Any:
        """Mark an object as tainted recursively with a set of origins."""
        for origin in origins:
            obj = mark_tainted(obj, origin)
        return obj

    def is_tainted(self, obj: Any) -> bool:
        """Check if an object or its contents are tainted."""
        return is_tainted(obj)

    def get_origins(self, obj: Any) -> Set[str]:
        """Get the origins of a tainted object."""
        return get_origins(obj)

    def sanitize(self, obj: Any) -> Any:
        """Sanitize an object recursively."""
        return sanitize(obj)


class SandboxExecutionEnvironmentV2:
    """A sandbox for executing tools with taint origin propagation."""
    def __init__(self, tracker: TaintTrackerV2) -> None:
        """Initialize the sandbox execution environment."""
        self.tracker = tracker

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function in the sandbox.
        Automatically propagates taint from inputs to output, aggregating origins.
        """
        origins = set()
        for arg in args:
            origins.update(self.tracker.get_origins(arg))
        for val in kwargs.values():
            origins.update(self.tracker.get_origins(val))

        # We pass sanitized arguments to the underlying function if needed,
        # but the standard implementation passes the objects as-is.
        result = func(*args, **kwargs)

        if origins:
            return self.tracker.taint_with_origins(result, origins)
        return result


class MCPKernelV2:
    """Kernel for executing tools safely with origin tracking."""
    def __init__(self) -> None:
        """Initialize the MCPKernelV2."""
        self.tracker = TaintTrackerV2()
        self.sandbox = SandboxExecutionEnvironmentV2(self.tracker)

    def execute_tool(self, tool_func: Callable[..., Any], inputs: Dict[str, Any], is_sensitive: bool = False) -> Any:
        """Executes a tool within the kernel. If is_sensitive is True, tainted inputs will fail."""
        if is_sensitive:
            if self.tracker.is_tainted(inputs):
                origins = self.tracker.get_origins(inputs)
                raise PolicyViolationError(f"Tainted input to sensitive tool call from origins: {origins}")

        try:
            result = self.sandbox.execute(tool_func, **inputs)
            return result
        except Exception as e:
            if isinstance(e, PolicyViolationError):
                raise
            raise RuntimeError(f"Tool execution failed: {str(e)}")
