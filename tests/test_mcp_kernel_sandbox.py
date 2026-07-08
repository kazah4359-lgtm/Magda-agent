"""Tests for MCP Kernel Taint Tracking Sandbox v1."""

import pytest

from magda_agent.safety.mcp_kernel_sandbox import MCPKernelSandbox
from magda_agent.safety.taint_tracking_v2 import PolicyViolationError, is_tainted, mark_tainted

def sample_sensitive_tool(data: str) -> str:
    """A sample sensitive tool."""
    return f"Processed {data}"

def sample_safe_tool(data: str) -> str:
    """A sample non-sensitive tool."""
    return f"Safe {data}"

def test_safe_inputs_accepted() -> None:
    """Test that a non-sensitive tool accepts safe data and returns untainted result."""
    sandbox = MCPKernelSandbox()

    safe_data = {"data": "alice"}

    result = sandbox.execute(sample_safe_tool, safe_data, is_sensitive=False)

    assert "Safe alice" in result
    assert not is_tainted(result)

def test_taint_propagation_non_sensitive() -> None:
    """Test that a non-sensitive tool accepts tainted data and propagates the taint."""
    sandbox = MCPKernelSandbox()

    tainted_string = sandbox.tracker.taint("DROP TABLE users;", "user_input")
    tainted_data = {"data": tainted_string}

    result = sandbox.execute(sample_safe_tool, tainted_data, is_sensitive=False)

    assert "Safe DROP TABLE users;" in result
    assert is_tainted(result)
    assert "user_input" in sandbox.tracker.get_origins(result)

def test_sensitive_tool_tainted_input() -> None:
    """Test that a sensitive tool rejects tainted data and raises PolicyViolationError."""
    sandbox = MCPKernelSandbox()

    tainted_string = sandbox.tracker.taint("DROP TABLE users;", "malicious_user")
    tainted_data = {"data": tainted_string}

    with pytest.raises(PolicyViolationError) as exc_info:
        sandbox.execute(sample_sensitive_tool, tainted_data, is_sensitive=True)

    assert "malicious_user" in str(exc_info.value)
    assert "Tainted input to sensitive tool call from origins" in str(exc_info.value)

def test_sensitive_tool_untainted_input() -> None:
    """Test that a sensitive tool accepts untainted data and returns untainted result."""
    sandbox = MCPKernelSandbox()

    safe_data = {"data": "alice"}

    result = sandbox.execute(sample_sensitive_tool, safe_data, is_sensitive=True)

    assert "Processed alice" in result
    assert not is_tainted(result)

def test_tool_execution_error() -> None:
    """Test that runtime errors in tool execution are wrapped in RuntimeError."""
    sandbox = MCPKernelSandbox()

    def failing_tool(data: str) -> str:
        raise ValueError("Tool failed")

    with pytest.raises(RuntimeError) as exc_info:
        sandbox.execute(failing_tool, {"data": "alice"}, is_sensitive=False)

    assert "Sandbox execution failed: Tool failed" in str(exc_info.value)
