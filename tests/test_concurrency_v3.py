import pytest
import asyncio
from magda_agent.skills.concurrency_v3 import ConcurrentSkillExecutor

class MockRegistry:
    def __init__(self):
        self.skills = {
            "sync_tool": self.sync_tool,
            "async_tool": self.async_tool,
        }

    def sync_tool(self, arg):
        import time
        time.sleep(0.1)
        return f"sync_{arg}"

    async def async_tool(self, arg):
        await asyncio.sleep(0.1)
        return f"async_{arg}"

    def execute_skill(self, name, **kwargs):
        return self.skills[name](**kwargs)

class MockMCPRegistry:
    def __init__(self):
        self.mcp_tools = {
            "mcp_async_tool": {"__is_async__": True, "name": "mcp_async_tool"}
        }

    def get_tool(self, name):
        return self.mcp_tools.get(name, {})

class MockRegistryWithMCP(MockRegistry):
    def __init__(self):
        super().__init__()
        self.skills["mcp_async_tool"] = self.mcp_async_tool

    def mcp_async_tool(self, arg): # Notice this is sync function but acts like an async due to how mcp client works (returns coroutine)
        async def inner():
            await asyncio.sleep(0.1)
            return f"mcp_{arg}"
        return inner()

@pytest.mark.asyncio
async def test_concurrent_execution_mixed():
    registry = MockRegistry()
    executor = ConcurrentSkillExecutor(registry)

    tool_calls = [
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
async def test_concurrent_execution_with_mcp_metadata():
    registry = MockRegistryWithMCP()
    mcp_registry = MockMCPRegistry()
    executor = ConcurrentSkillExecutor(registry, mcp_registry)

    tool_calls = [
        {"name": "mcp_async_tool", "kwargs": {"arg": "1"}},
    ]

    results = await executor.execute_concurrently(tool_calls)
    assert results == ["mcp_1"]


@pytest.mark.asyncio
async def test_tool_not_found():
    registry = MockRegistry()
    executor = ConcurrentSkillExecutor(registry)

    tool_calls = [
        {"name": "missing_tool", "kwargs": {}},
    ]

    results = await executor.execute_concurrently(tool_calls)
    assert results[0] == "Error: Skill 'missing_tool' not found."
