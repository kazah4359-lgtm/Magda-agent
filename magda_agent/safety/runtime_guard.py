from typing import Callable, Any
from magda_agent.security.mcp_kernel_taint import is_tainted

class SecurityException(Exception):
    """Exception raised for security violations."""
    pass

class RuntimeGuard:
    """Runtime guard for executing tools safely."""

    @staticmethod
    def execute_safely(tool_func: Callable, **kwargs: Any) -> Any:
        """
        Executes a tool function safely, checking for tainted arguments.

        Raises:
            SecurityException: If any argument is tainted.
        """
        for k, v in kwargs.items():
            if isinstance(v, str) and is_tainted(v):
                raise SecurityException(f"Argument '{k}' is tainted and unsafe to execute.")
        return tool_func(**kwargs)
