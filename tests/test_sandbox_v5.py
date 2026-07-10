"""Tests for MCP Kernel Taint Tracking Sandbox v5."""

import asyncio
import pytest

from magda_agent.safety.sandbox_v5 import MCPKernelSandboxV5, ContainerIsolationError
from magda_agent.safety.taint_tracking_v2 import PolicyViolationError, is_tainted

def sample_sync_tool(data: str) -> str:
    """A sample synchronous tool."""
    return f"Processed {data}"

async def sample_async_tool(data: str) -> str:
    """A sample asynchronous tool."""
    return f"Processed Async {data}"

async def slow_async_tool(data: str) -> str:
    """A sample slow asynchronous tool that sleeps."""
    await asyncio.sleep(0.5)
    return f"Processed Slow {data}"

def test_sync_safe_inputs_accepted() -> None:
    """Test that a synchronous tool accepts safe data and returns untainted result."""
    sandbox = MCPKernelSandboxV5()
    result = sandbox.execute(sample_sync_tool, {"data": "alice"}, is_sensitive=False)
    assert "Processed alice" in result
    assert not is_tainted(result)

def test_sync_taint_propagation_non_sensitive() -> None:
    """Test that a synchronous tool propagates taint when not sensitive."""
    sandbox = MCPKernelSandboxV5()
    tainted_string = sandbox.tracker.taint("DROP TABLE users;", "user_input")

    result = sandbox.execute(sample_sync_tool, {"data": tainted_string}, is_sensitive=False)

    assert "Processed DROP TABLE users;" in result
    assert is_tainted(result)
    assert "user_input" in sandbox.tracker.get_origins(result)

def test_sync_sensitive_tool_tainted_input() -> None:
    """Test that a sensitive sync tool rejects tainted data."""
    sandbox = MCPKernelSandboxV5()
    tainted_string = sandbox.tracker.taint("DROP TABLE", "malicious")

    with pytest.raises(PolicyViolationError) as exc_info:
        sandbox.execute(sample_sync_tool, {"data": tainted_string}, is_sensitive=True)

    assert "malicious" in str(exc_info.value)

def test_sync_execution_error_handling() -> None:
    """Test that sync execution errors are wrapped in RuntimeError."""
    sandbox = MCPKernelSandboxV5()
    def failing_tool(data: str) -> str:
        raise ValueError("Tool failed")

    with pytest.raises(RuntimeError) as exc_info:
        sandbox.execute(failing_tool, {"data": "alice"}, is_sensitive=False)

    assert "Sandbox execution failed: Tool failed" in str(exc_info.value)

def test_sync_rejects_async_tool() -> None:
    """Test that execute() rejects async coroutine functions."""
    sandbox = MCPKernelSandboxV5()
    with pytest.raises(ValueError) as exc_info:
        sandbox.execute(sample_async_tool, {"data": "alice"})
    assert "use execute_async" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_async_safe_inputs_accepted() -> None:
    """Test that an async tool accepts safe data and returns untainted result."""
    sandbox = MCPKernelSandboxV5()
    result = await sandbox.execute_async(sample_async_tool, {"data": "bob"}, is_sensitive=False)
    assert "Processed Async bob" in result
    assert not is_tainted(result)

@pytest.mark.asyncio
async def test_async_taint_propagation() -> None:
    """Test that an async tool propagates taint when not sensitive."""
    sandbox = MCPKernelSandboxV5()
    tainted_string = sandbox.tracker.taint("SELECT *", "db_input")

    result = await sandbox.execute_async(sample_async_tool, {"data": tainted_string}, is_sensitive=False)

    assert "Processed Async SELECT *" in result
    assert is_tainted(result)
    assert "db_input" in sandbox.tracker.get_origins(result)

@pytest.mark.asyncio
async def test_async_sensitive_tool_tainted_input() -> None:
    """Test that a sensitive async tool rejects tainted data."""
    sandbox = MCPKernelSandboxV5()
    tainted_string = sandbox.tracker.taint("DELETE", "hacker")

    with pytest.raises(PolicyViolationError):
        await sandbox.execute_async(sample_async_tool, {"data": tainted_string}, is_sensitive=True)

@pytest.mark.asyncio
async def test_async_timeout_enforcement() -> None:
    """Test that slow async tools timeout due to container isolation limits."""
    sandbox = MCPKernelSandboxV5(timeout_seconds=0.1) # 100ms timeout

    with pytest.raises(ContainerIsolationError) as exc_info:
        await sandbox.execute_async(slow_async_tool, {"data": "slowpoke"}, is_sensitive=False)

    assert "timed out" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_async_execution_error_handling() -> None:
    """Test that async execution errors are wrapped in RuntimeError."""
    sandbox = MCPKernelSandboxV5()
    async def failing_async_tool(data: str) -> str:
        raise TypeError("Async crash")

    with pytest.raises(RuntimeError) as exc_info:
        await sandbox.execute_async(failing_async_tool, {"data": "alice"}, is_sensitive=False)

    assert "Sandbox async execution failed: Async crash" in str(exc_info.value)

@pytest.mark.asyncio
async def test_async_rejects_sync_tool() -> None:
    """Test that execute_async() rejects sync functions."""
    sandbox = MCPKernelSandboxV5()
    with pytest.raises(ValueError) as exc_info:
        await sandbox.execute_async(sample_sync_tool, {"data": "alice"})
    assert "requires a coroutine function" in str(exc_info.value).lower()
