from magda_agent.security.mcp_kernel import MCPKernel, SecurityError
from typing import Dict, Any

def execute(code: str) -> str:
    """
    Executes code using the MCPKernel sandbox with taint tracking.
    """
    kernel = MCPKernel()
    try:
        locals_dict: Dict[str, Any] = {}
        kernel.execute(code, locals_dict=locals_dict)
        return str(locals_dict)
    except SecurityError as e:
        return f"SecurityError: {e}"
    except Exception as e:
        return f"ExecutionError: {e}"
