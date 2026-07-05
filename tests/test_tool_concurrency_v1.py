import pytest
import asyncio
from typing import Any, Dict, List
from magda_agent.skills.tool_concurrency_v1 import ConcurrentToolExecutorV1

class MockRegistry:
    """Mock skill registry for testing."""
    def __init__(self) -> None:
        """Initialize mock registry with sync and async tools."""
        self.skills: Dict[str, Any] = {
            "sync_tool": self.sync_tool,
            "async_tool": self.async_tool,
        }

    def sync_tool(self, arg: str) -> str:
        """A synchronous mock tool."""
        import time
        time.sleep(0.1)
        return f"sync_{arg}"

    async def async_tool(self, arg: str) -> str:
        """An asynchronous mock tool."""
        await asyncio.sleep(0.1)
        return f"async_{arg}"

    def execute_skill(self, name: str, **kwargs: Any) -> Any:
        """Execute a skill from the registry."""
        return self.skills[name](**kwargs)

class MockMCPRegistry:
    """Mock MCP registry for testing."""
    def __init__(self) -> None:
        """Initialize mock MCP registry with a tool marked as async."""
        self.mcp_tools: Dict[str, Dict[str, Any]] = {
            "mcp_async_tool": {"__is_async__": True, "name": "mcp_async_tool"}
        }

    def get_tool(self, name: str) -> Dict[str, Any]:
        """Get tool metadata by name."""
        return self.mcp_tools.get(name, {})

class MockRegistryWithMCP(MockRegistry):
    """Mock registry that includes an MCP tool."""
    def __init__(self) -> None:
        """Initialize mock registry with MCP tool."""
        super().__init__()
        self.skills["mcp_async_tool"] = self.mcp_async_tool

    def mcp_async_tool(self, arg: str) -> Any:
        """An MCP tool that returns a coroutine."""
        async def inner() -> str:
            """Inner coroutine."""
            await asyncio.sleep(0.1)
            return f"mcp_{arg}"
        return inner()

@pytest.mark.asyncio
async def test_concurrent_execution_mixed() -> None:
    """Test concurrent execution of mixed sync and async tools."""
    registry = MockRegistry()
    executor = ConcurrentToolExecutorV1(registry)

    tool_calls: List[Dict[str, Any]] = [
        {"name": "sync_tool", "kwargs": {"arg": "1"}},
        {"name": "async_tool", "kwargs": {"arg": "2"}},
        {"name": "sync_tool", "kwargs": {"arg": "3"}},
    ]

    import time
    start = time.time()
    results = await executor.execute_concurrently(tool_calls)
    end = time.time()

    assert results == ["sync_1", "async_2", "sync_3"]
    # All should take ~0.1s concurrently
    assert end - start < 0.2

@pytest.mark.asyncio
async def test_concurrent_execution_with_mcp_metadata() -> None:
    """Test concurrent execution using MCP tool metadata."""
    registry = MockRegistryWithMCP()
    mcp_registry = MockMCPRegistry()
    executor = ConcurrentToolExecutorV1(registry, mcp_registry)

    tool_calls: List[Dict[str, Any]] = [
        {"name": "mcp_async_tool", "kwargs": {"arg": "1"}},
    ]

    results = await executor.execute_concurrently(tool_calls)
    assert results == ["mcp_1"]


@pytest.mark.asyncio
async def test_tool_not_found() -> None:
    """Test execution when a tool is not found."""
    registry = MockRegistry()
    executor = ConcurrentToolExecutorV1(registry)

    tool_calls: List[Dict[str, Any]] = [
        {"name": "missing_tool", "kwargs": {}},
    ]

    results = await executor.execute_concurrently(tool_calls)
    assert results[0] == "Error: Skill 'missing_tool' not found."
