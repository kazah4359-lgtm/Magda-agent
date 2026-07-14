"""Tests for MCPKernel Taint Tracking V2 and TaintTrackingAgentGuard."""
import pytest
from unittest.mock import MagicMock

from magda_agent.safety.agent_guard import SecurityViolationError
from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.taint_tracking_v2 import (
    mark_tainted,
    is_tainted,
    get_origins,
    sanitize,
    TaintTrackerV2,
    SandboxExecutionEnvironmentV2,
    MCPKernelV2,
    PolicyViolationError,
    TaintedString,
    TaintTrackingAgentGuard
)


def test_mark_tainted_and_is_tainted() -> None:
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


def test_sanitize() -> None:
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


def test_sandbox_execution_environment() -> None:
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


def test_mcp_kernel_v2_policy_violation() -> None:
    """Test MCPKernelV2 policy violation on sensitive operations."""
    kernel = MCPKernelV2()

    def mock_sensitive_tool(data: str) -> str:
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


def test_mcp_kernel_v2_execution_error() -> None:
    """Test MCPKernelV2 handles regular exceptions properly."""
    kernel = MCPKernelV2()

    def mock_failing_tool(data: str) -> None:
        raise ValueError("Tool crashed")

    with pytest.raises(RuntimeError, match="Tool execution failed: Tool crashed"):
        kernel.execute_tool(mock_failing_tool, {"data": "test"})


def test_taint_tracking_agent_guard_sync() -> None:
    """Test TaintTrackingAgentGuard with synchronous tool and taint propagation."""
    policy_mock = MagicMock(spec=PolicyLayer)
    policy_mock.evaluate.return_value = (True, "Allowed")
    guard = TaintTrackingAgentGuard(policy_layer=policy_mock)

    # Taint an input entering context
    external_input = guard.taint_input("unsafe_command", "external_api")
    assert guard.tracker.is_tainted(external_input)
    assert "external_api" in guard.tracker.get_origins(external_input)

    def mock_sync_tool(cmd: str) -> str:
        return f"run {cmd}"

    # Execute tool
    result = guard.execute_tool(mock_sync_tool, "test_tool", cmd=external_input)

    # Assert taint propagates to output
    assert guard.tracker.is_tainted(result)
    assert "external_api" in guard.tracker.get_origins(result)
    assert result == "run unsafe_command"


@pytest.mark.asyncio
async def test_taint_tracking_agent_guard_async() -> None:
    """Test TaintTrackingAgentGuard with asynchronous tool and taint propagation."""
    policy_mock = MagicMock(spec=PolicyLayer)
    policy_mock.evaluate.return_value = (True, "Allowed")
    guard = TaintTrackingAgentGuard(policy_layer=policy_mock)

    # Taint an input entering context
    external_input = guard.taint_input("unsafe_query", "webhook")
    assert guard.tracker.is_tainted(external_input)
    assert "webhook" in guard.tracker.get_origins(external_input)

    async def mock_async_tool(query: str) -> str:
        return f"query {query}"

    # Execute async tool
    coro = guard.execute_tool(mock_async_tool, "test_async_tool", query=external_input)
    result = await coro

    # Assert taint propagates to output
    assert guard.tracker.is_tainted(result)
    assert "webhook" in guard.tracker.get_origins(result)
    assert result == "query unsafe_query"


def test_taint_tracking_agent_guard_sensitive_block() -> None:
    """Test TaintTrackingAgentGuard blocks sensitive tool calls with tainted inputs."""
    policy_mock = MagicMock(spec=PolicyLayer)
    policy_mock.evaluate.return_value = (True, "Allowed by policy")
    guard = TaintTrackingAgentGuard(
        policy_layer=policy_mock,
        sensitive_tools={"dangerous_tool"}
    )

    external_input = guard.taint_input("dangerous_param", "untrusted_user")

    def dangerous_tool(param: str) -> str:
        return f"destroy {param}"

    # Verify calling dangerous_tool with tainted input raises SecurityViolationError
    with pytest.raises(SecurityViolationError) as excinfo:
        guard.execute_tool(dangerous_tool, "dangerous_tool", param=external_input)

    assert "untrusted_user" in str(excinfo.value)
    assert "dangerous_tool" in str(excinfo.value)
