"""MCPKernel Taint Tracking Sandbox V2.

Provides taint tracking with origin propagation and sandboxing for tool execution,
and integrates with the Agent Guard runtime policy layer.
"""
import inspect
import logging
from typing import Any, Callable, Dict, List, Set, Union

from magda_agent.safety.agent_guard import AgentGuard, SecurityViolationError


class PolicyViolationError(Exception):
    """Raised when tainted data violates policy."""
    pass


class TaintedData:
    """Base class for tainted data wrappers that track origins."""
    def __init__(self, value: Any, origins: Set[str] = None) -> None:
        """Initialize the TaintedData.

        Args:
            value: The underlying untainted value.
            origins: Optional set of origin sources.
        """
        self.value = value
        self.origins = origins if origins is not None else set()


class TaintedString(TaintedData, str):
    """A string subclass that marks data as tainted and tracks its origins."""
    def __new__(cls, value: str, origins: Set[str] = None) -> "TaintedString":
        """Create a new TaintedString instance."""
        obj = str.__new__(cls, value)
        return obj

    def __init__(self, value: str, origins: Set[str] = None) -> None:
        """Initialize the TaintedString.

        Args:
            value: The string value.
            origins: Optional set of origin sources.
        """
        TaintedData.__init__(self, value, origins)


def mark_tainted(s: Any, origin: str) -> Any:
    """Marks a string, list, or dict as tainted recursively with the given origin.

    Args:
        s: The object to taint.
        origin: The source origin.

    Returns:
        The tainted object.
    """
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
    """Checks if an object or any of its nested structures is tainted.

    Args:
        s: The object to check.

    Returns:
        True if tainted, False otherwise.
    """
    if isinstance(s, TaintedData):
        return True
    if isinstance(s, list):
        return any(is_tainted(item) for item in s)
    if isinstance(s, dict):
        return any(is_tainted(k) or is_tainted(v) for k, v in s.items())
    return False


def get_origins(s: Any) -> Set[str]:
    """Retrieves all origins from a tainted object recursively.

    Args:
        s: The object to inspect.

    Returns:
        A set of all distinct origin strings.
    """
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
    """Sanitizes tainted data, returning regular primitives recursively.

    Args:
        s: The object to sanitize.

    Returns:
        The untainted, sanitized object.
    """
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
        """Mark an object as tainted recursively with an origin.

        Args:
            obj: The object to taint.
            origin: The origin source.

        Returns:
            The tainted object.
        """
        return mark_tainted(obj, origin)

    def taint_with_origins(self, obj: Any, origins: Set[str]) -> Any:
        """Mark an object as tainted recursively with a set of origins.

        Args:
            obj: The object to taint.
            origins: The set of origins.

        Returns:
            The tainted object.
        """
        for origin in origins:
            obj = mark_tainted(obj, origin)
        return obj

    def is_tainted(self, obj: Any) -> bool:
        """Check if an object or its contents are tainted.

        Args:
            obj: The object to check.

        Returns:
            True if tainted, False otherwise.
        """
        return is_tainted(obj)

    def get_origins(self, obj: Any) -> Set[str]:
        """Get the origins of a tainted object.

        Args:
            obj: The object to inspect.

        Returns:
            The set of origins.
        """
        return get_origins(obj)

    def sanitize(self, obj: Any) -> Any:
        """Sanitize an object recursively.

        Args:
            obj: The object to sanitize.

        Returns:
            The sanitized object.
        """
        return sanitize(obj)


class SandboxExecutionEnvironmentV2:
    """A sandbox for executing tools with taint origin propagation."""
    def __init__(self, tracker: TaintTrackerV2) -> None:
        """Initialize the sandbox execution environment.

        Args:
            tracker: The TaintTrackerV2 instance.
        """
        self.tracker = tracker

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a function in the sandbox.

        Automatically propagates taint from inputs to output, aggregating origins.

        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of execution, tainted if any inputs were tainted.
        """
        origins = set()
        for arg in args:
            origins.update(self.tracker.get_origins(arg))
        for val in kwargs.values():
            origins.update(self.tracker.get_origins(val))

        result = func(*args, **kwargs)

        if inspect.isawaitable(result):
            async def async_wrapper() -> Any:
                res = await result
                if origins:
                    return self.tracker.taint_with_origins(res, origins)
                return res
            return async_wrapper()

        if origins:
            return self.tracker.taint_with_origins(result, origins)
        return result

    async def execute_async(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute an async function in the sandbox.

        Automatically propagates taint from inputs to output, aggregating origins.

        Args:
            func: The async function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The awaited result of execution, tainted if any inputs were tainted.
        """
        origins = set()
        for arg in args:
            origins.update(self.tracker.get_origins(arg))
        for val in kwargs.values():
            origins.update(self.tracker.get_origins(val))

        result = await func(*args, **kwargs)

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
        """Executes a tool within the kernel. If is_sensitive is True, tainted inputs will fail.

        Args:
            tool_func: The tool function.
            inputs: Dict of input arguments.
            is_sensitive: Whether the tool is sensitive and should block tainted inputs.

        Returns:
            The result of tool execution.
        """
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


class TaintTrackingAgentGuard(AgentGuard):
    """An enhanced Agent Guard that implements taint tracking for external data traversing the policy layer."""
    def __init__(self, policy_layer: Any, tracker: TaintTrackerV2 = None, sensitive_tools: Set[str] = None) -> None:
        """Initializes the TaintTrackingAgentGuard.

        Args:
            policy_layer: The policy layer used to evaluate actions.
            tracker: Optional TaintTrackerV2 instance.
            sensitive_tools: Optional set of tool names considered sensitive.
        """
        super().__init__(policy_layer)
        self.tracker = tracker or TaintTrackerV2()
        self.sandbox = SandboxExecutionEnvironmentV2(self.tracker)
        self.sensitive_tools = set(sensitive_tools) if sensitive_tools is not None else set()

    def taint_input(self, data: Any, origin: str) -> Any:
        """Marks an external input as tainted when it enters the agent's context.

        Args:
            data: The external input data to taint.
            origin: The origin source of the taint.

        Returns:
            The recursively tainted data.
        """
        return self.tracker.taint(data, origin)

    def execute_tool(self, tool_func: Callable, tool_name: str, **kwargs: Any) -> Any:
        """Intercepts and evaluates a tool call before executing it, ensuring taint propagation.

        Args:
            tool_func: The actual tool function to execute if permitted.
            tool_name: The name of the tool/action to evaluate.
            **kwargs: The arguments to pass to the tool.

        Returns:
            The result of the tool execution with propagated taints.

        Raises:
            SecurityViolationError: If action is blocked by policy or is sensitive with tainted input.
        """
        # If the tool is sensitive and has tainted input, block it immediately
        if tool_name in self.sensitive_tools:
            if self.tracker.is_tainted(kwargs):
                origins = self.tracker.get_origins(kwargs)
                explanation = f"Tainted input to sensitive tool '{tool_name}' from origins: {origins}"
                self.logger.warning(
                    f"AgentGuard: Tool execution blocked for '{tool_name}'. Reason: {explanation}"
                )
                raise SecurityViolationError(f"Action '{tool_name}' blocked: {explanation}")

        # Regular policy layer evaluation
        allow, explanation = self.policy_layer.evaluate(tool_name, **kwargs)

        if not allow:
            self.logger.warning(
                f"AgentGuard: Tool execution blocked for '{tool_name}'. Reason: {explanation}"
            )
            raise SecurityViolationError(f"Action '{tool_name}' blocked: {explanation}")

        self.logger.info(f"AgentGuard: Tool execution permitted for '{tool_name}'.")

        # Execute in sandbox environment
        if inspect.iscoroutinefunction(tool_func):
            async def async_wrapper() -> Any:
                return await self.sandbox.execute_async(tool_func, **kwargs)
            return async_wrapper()

        result = self.sandbox.execute(tool_func, **kwargs)
        return result
