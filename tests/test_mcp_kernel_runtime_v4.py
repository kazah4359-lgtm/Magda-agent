"""Tests for the MCPKernelRuntimeV4 module."""

import pytest

from magda_agent.safety.mcp_kernel_runtime_v4 import MCPKernelRuntimeV4
from magda_agent.safety.mcp_taint_policy import MCPTaintPolicyEngine
from magda_agent.safety.taint_tracking_v2 import PolicyViolationError, TaintTrackerV2


@pytest.fixture
def runtime() -> MCPKernelRuntimeV4:
    """Fixture providing an MCPKernelRuntimeV4 instance."""
    return MCPKernelRuntimeV4()


def test_execute_tool_sensitive_rejects_tainted_input(runtime: MCPKernelRuntimeV4) -> None:
    """Test that a sensitive tool rejects tainted input."""
    def sample_tool(data: str) -> str:
        return f"Processed {data}"

    tracker = runtime.policy_engine.tracker
    tainted_input = tracker.taint({"data": "untrusted"}, "malicious_source")

    with pytest.raises(PolicyViolationError, match="Tainted input to sensitive tool call"):
        runtime.execute_tool(sample_tool, inputs=tainted_input, is_sensitive=True)


def test_execute_tool_non_sensitive_accepts_tainted_input(runtime: MCPKernelRuntimeV4) -> None:
    """Test that a non-sensitive tool accepts tainted input."""
    def sample_tool(data: str) -> str:
        return f"Processed {data}"

    tracker = runtime.policy_engine.tracker
    # Input is tainted but not the string itself to bypass simple tests, let's taint a string.
    tainted_str = tracker.taint("untrusted", "malicious_source")

    # We execute and return the tainted string
    def sample_tool_2(data: str) -> str:
        return data

    inputs = {"data": tainted_str}

    # Tool is not sensitive and output is untrusted. We should fail output validation
    # since output is tainted and tool is untrusted.
    with pytest.raises(PolicyViolationError, match="Tainted output from untrusted tool call"):
         runtime.execute_tool(sample_tool_2, inputs=inputs, is_sensitive=False, is_trusted=False)


def test_execute_tool_untrusted_rejects_tainted_output(runtime: MCPKernelRuntimeV4) -> None:
    """Test that an untrusted tool rejects tainted output."""
    def sample_tool() -> str:
        return runtime.policy_engine.tracker.taint("secret", "db")

    with pytest.raises(PolicyViolationError, match="Tainted output from untrusted tool call"):
        runtime.execute_tool(sample_tool, inputs={}, is_sensitive=False, is_trusted=False)


def test_execute_tool_trusted_accepts_tainted_output(runtime: MCPKernelRuntimeV4) -> None:
    """Test that a trusted tool accepts tainted output."""
    def sample_tool() -> str:
        return runtime.policy_engine.tracker.taint("secret", "db")

    result = runtime.execute_tool(sample_tool, inputs={}, is_sensitive=False, is_trusted=True)
    assert runtime.policy_engine.tracker.is_tainted(result)


def test_runtime_proxy_sync(runtime: MCPKernelRuntimeV4) -> None:
    """Test the runtime_proxy decorator with a synchronous function."""

    @runtime.runtime_proxy(is_sensitive=True, is_trusted=False)
    def my_sync_tool(a: str, b: str = "default") -> str:
        return f"{a} {b}"

    # Clean inputs -> clean output
    res = my_sync_tool("hello")
    assert res == "hello default"

    # Tainted input -> rejected because sensitive
    tainted_input = runtime.policy_engine.tracker.taint("bad", "source")
    with pytest.raises(PolicyViolationError, match="Tainted input to sensitive tool call"):
        my_sync_tool(tainted_input)

    @runtime.runtime_proxy(is_sensitive=False, is_trusted=False)
    def my_tainted_tool() -> str:
         return runtime.policy_engine.tracker.taint("secret", "db")

    # Tainted output -> rejected because untrusted
    with pytest.raises(PolicyViolationError, match="Tainted output from untrusted tool call"):
        my_tainted_tool()


@pytest.mark.asyncio
async def test_runtime_proxy_async(runtime: MCPKernelRuntimeV4) -> None:
    """Test the runtime_proxy decorator with an asynchronous function."""

    @runtime.runtime_proxy(is_sensitive=True, is_trusted=False)
    async def my_async_tool(a: str, b: str = "default") -> str:
        return f"{a} {b}"

    # Clean inputs -> clean output
    res = await my_async_tool("hello")
    assert res == "hello default"

    # Tainted input -> rejected because sensitive
    tainted_input = runtime.policy_engine.tracker.taint("bad", "source")
    with pytest.raises(PolicyViolationError, match="Tainted input to sensitive tool call"):
        await my_async_tool(tainted_input)

    @runtime.runtime_proxy(is_sensitive=False, is_trusted=True)
    async def my_trusted_async_tool() -> str:
         return runtime.policy_engine.tracker.taint("secret", "db")

    # Tainted output -> accepted because trusted
    res2 = await my_trusted_async_tool()
    assert runtime.policy_engine.tracker.is_tainted(res2)


@pytest.mark.asyncio
async def test_execute_tool_async(runtime: MCPKernelRuntimeV4) -> None:
    """Test execute_tool directly with an asynchronous function."""
    async def sample_tool() -> str:
        return "async clean"

    res = await runtime.execute_tool(sample_tool, inputs={}, is_sensitive=False, is_trusted=False)
    assert res == "async clean"
