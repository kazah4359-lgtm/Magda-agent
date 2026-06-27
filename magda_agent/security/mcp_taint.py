"""MCP Tool Taint Tracking Sandbox v2."""
import inspect
import functools
from typing import Any, Callable, List, Optional
from magda_agent.security.mcp_kernel_taint import is_tainted, mark_tainted, PolicyViolationError

class TaintSandboxError(PolicyViolationError):
    """Raised when tainted data violates policy during MCP tool execution."""
    pass

def mcp_action_taint_sandbox(critical_params: Optional[List[str]] = None) -> Callable:
    """
    Decorator for MCP action tools to track taint and block unsafe calls.

    Args:
        critical_params: List of parameter names that must not receive tainted data.
    """
    if critical_params is None:
        critical_params = []

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Map args and kwargs to parameter names
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Block if any critical parameter receives tainted data
            for param_name, param_value in bound_args.arguments.items():
                if param_name in critical_params and is_tainted(param_value):
                    raise TaintSandboxError(
                        f"Critical parameter '{param_name}' in '{func.__name__}' received tainted data."
                    )

            # Execute the function
            result = func(*args, **kwargs)

            # Outputs from external MCP action tools are inherently untrusted and thus tainted
            return mark_tainted(result)

        return wrapper
    return decorator
