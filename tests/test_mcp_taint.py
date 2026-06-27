import pytest
from magda_agent.security.mcp_taint import mcp_action_taint_sandbox, TaintSandboxError
from magda_agent.security.mcp_kernel_taint import mark_tainted, is_tainted

def test_mcp_action_taint_sandbox_blocks_critical_param() -> None:
    """Tests that the sandbox blocks a critical parameter if it is tainted."""
    @mcp_action_taint_sandbox(critical_params=["command"])
    def execute_command(command: str, background: bool = False) -> str:
        return f"Executed {command}"

    tainted_command = mark_tainted("rm -rf /")

    with pytest.raises(TaintSandboxError, match="Critical parameter 'command' in 'execute_command' received tainted data."):
        execute_command(tainted_command)

def test_mcp_action_taint_sandbox_allows_clean_param() -> None:
    """Tests that the sandbox allows a clean critical parameter."""
    @mcp_action_taint_sandbox(critical_params=["command"])
    def execute_command(command: str, background: bool = False) -> str:
        return f"Executed {command}"

    clean_command = "ls -l"
    result = execute_command(clean_command)
    assert result == "Executed ls -l"

def test_mcp_action_taint_sandbox_taints_output() -> None:
    """Tests that the sandbox automatically taints the output of the wrapped function."""
    @mcp_action_taint_sandbox(critical_params=["query"])
    def search_web(query: str) -> dict:
        return {"results": [f"Result for {query}"]}

    clean_query = "weather"
    result = search_web(clean_query)

    assert is_tainted(result) is True
    assert result == {"results": ["Result for weather"]}

def test_mcp_action_taint_sandbox_allows_taint_on_non_critical_param() -> None:
    """Tests that the sandbox allows tainted data on non-critical parameters."""
    @mcp_action_taint_sandbox(critical_params=["query"])
    def search_web(query: str, extra_info: str = "") -> dict:
        return {"results": [f"Result for {query} with {extra_info}"]}

    clean_query = "weather"
    tainted_extra = mark_tainted("bad_data")

    result = search_web(clean_query, extra_info=tainted_extra)
    assert is_tainted(result) is True
    assert result == {"results": [f"Result for {clean_query} with {tainted_extra}"]}
