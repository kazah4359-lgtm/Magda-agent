"""Tests for MCPKernel Taint Tracking V2."""
import pytest
from magda_agent.safety.taint_tracking_v2 import (
    mark_tainted,
    is_tainted,
    get_origins,
    sanitize,
    TaintTrackerV2,
    SandboxExecutionEnvironmentV2,
    MCPKernelV2,
    PolicyViolationError,
    TaintedString
)


def test_mark_tainted_and_is_tainted():
    """Test marking primitives and structures as tainted."""
    # Primitive string
    s = mark_tainted("hello", "user_input")
    assert is_tainted(s)
    assert "user_input" in get_origins(s)
    assert isinstance(s, TaintedString)
    assert s == "hello"

    # Multiple taints
    s = mark_tainted(s, "db_input")
    assert "user_input" in get_origins(s)
    assert "db_input" in get_origins(s)

    # List
    l = mark_tainted(["a", "b"], "list_origin")
    assert is_tainted(l)
    assert "list_origin" in get_origins(l)
    assert is_tainted(l[0])

    # Dict
    d = mark_tainted({"key": "value"}, "dict_origin")
    assert is_tainted(d)
    assert "dict_origin" in get_origins(d)
    # The value should be tainted
    assert is_tainted(d["key"])


def test_sanitize():
    """Test sanitizing tainted data."""
    s = mark_tainted("hello", "user_input")
    sanitized_s = sanitize(s)
    assert not is_tainted(sanitized_s)
    assert not isinstance(sanitized_s, TaintedString)
    assert sanitized_s == "hello"

    d = mark_tainted({"k": ["v1", "v2"]}, "dict_origin")
    sanitized_d = sanitize(d)
    assert not is_tainted(sanitized_d)
    assert isinstance(sanitized_d["k"][0], str)
    assert not isinstance(sanitized_d["k"][0], TaintedString)


def test_sandbox_execution_environment():
    """Test origin propagation through sandbox execution."""
    tracker = TaintTrackerV2()
    sandbox = SandboxExecutionEnvironmentV2(tracker)

    def my_tool(a: str, b: str) -> str:
        """A simple mock tool."""
        return a + b

    # Test with no taints
    res1 = sandbox.execute(my_tool, "a", "b")
    assert not is_tainted(res1)

    # Test with tainted args
    tainted_a = tracker.taint("a", "source_a")
    res2 = sandbox.execute(my_tool, tainted_a, "b")
    assert is_tainted(res2)
    assert "source_a" in tracker.get_origins(res2)

    # Test with multiple tainted args and kwargs
    tainted_b = tracker.taint("b", "source_b")
    res3 = sandbox.execute(my_tool, tainted_a, b=tainted_b)
    assert is_tainted(res3)
    origins = tracker.get_origins(res3)
    assert "source_a" in origins
    assert "source_b" in origins


def test_mcp_kernel_v2_policy_violation():
    """Test MCPKernelV2 policy violation on sensitive operations."""
    kernel = MCPKernelV2()

    def mock_sensitive_tool(data: str):
        """Mock sensitive tool."""
        return f"Executed: {data}"

    tainted_input = kernel.tracker.taint("secret_command", "malicious_user")

    # Non-sensitive execution should pass and propagate taint
    res = kernel.execute_tool(mock_sensitive_tool, {"data": tainted_input}, is_sensitive=False)
    assert kernel.tracker.is_tainted(res)
    assert "malicious_user" in kernel.tracker.get_origins(res)

    # Sensitive execution should raise PolicyViolationError
    with pytest.raises(PolicyViolationError) as excinfo:
        kernel.execute_tool(mock_sensitive_tool, {"data": tainted_input}, is_sensitive=True)

    assert "malicious_user" in str(excinfo.value)
    assert "Tainted input to sensitive tool call from origins" in str(excinfo.value)


def test_mcp_kernel_v2_execution_error():
    """Test MCPKernelV2 handles regular exceptions properly."""
    kernel = MCPKernelV2()

    def mock_failing_tool(data: str):
        raise ValueError("Tool crashed")

    with pytest.raises(RuntimeError, match="Tool execution failed: Tool crashed"):
        kernel.execute_tool(mock_failing_tool, {"data": "test"})
