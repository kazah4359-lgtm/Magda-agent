import asyncio
import time
import inspect
import pytest
from typing import Any, Dict, List
from magda_agent.skills.concurrency_v5 import ConcurrentToolRouterV5
from magda_agent.skills.registry import SkillRegistry
from magda_agent.integration.mcp_server import MCPServer
from magda_agent.integration.mcp_exporter import MCPExporter


class MockServerRegistry:
    """Mock registry class resembling standard SkillRegistry."""

    def __init__(self) -> None:
        self.skills = {
            "test_skill": self.test_skill,
            "async_test_skill": self.async_test_skill,
        }

    def test_skill(self, val: str) -> str:
        return f"result_sync_{val}"

    async def async_test_skill(self, val: str) -> str:
        await asyncio.sleep(0.05)
        return f"result_async_{val}"

    async def execute_skill(self, name: str, **kwargs: Any) -> Any:
        res = self.skills[name](**kwargs)
        if inspect.isawaitable(res):
            return await res
        return res


@pytest.fixture
def local_registry() -> SkillRegistry:
    """Provides a local SkillRegistry with dummy tools."""
    reg = SkillRegistry()

    def local_sync(val: str) -> str:
        time.sleep(0.01)
        return f"local_sync_{val}"

    async def local_async(val: str) -> str:
        await asyncio.sleep(0.01)
        return f"local_async_{val}"

    reg.register_skill("local_sync", local_sync, "A local sync skill")
    reg.register_skill("local_async", local_async, "A local async skill")
    return reg


@pytest.mark.asyncio
async def test_router_local_only(local_registry: SkillRegistry) -> None:
    """Tests executing local tools only (no prefixes)."""
    router = ConcurrentToolRouterV5(local_registry)

    tool_calls = [
        {"name": "local_sync", "kwargs": {"val": "1"}},
        {"name": "local_async", "kwargs": {"val": "2"}},
    ]

    start = time.time()
    results = await router.execute_concurrently(tool_calls)
    end = time.time()

    assert results == ["local_sync_1", "local_async_2"]
    # Should run in parallel. A generous limit is set to avoid CI CPU scheduling flakiness.
    assert end - start < 3.0


@pytest.mark.asyncio
async def test_router_prefixed_routing(local_registry: SkillRegistry) -> None:
    """Tests routing tools to prefix-matched servers with multiple separators."""
    server1 = MockServerRegistry()
    server2 = MockServerRegistry()

    servers = {
        "srv1": server1,
        "srv2": server2,
    }

    router = ConcurrentToolRouterV5(
        registry=local_registry,
        servers=servers,
        separators=["__", "-", "_"]
    )

    tool_calls = [
        {"name": "srv1__test_skill", "kwargs": {"val": "a"}},
        {"name": "srv2-async_test_skill", "kwargs": {"val": "b"}},
        {"name": "srv1_test_skill", "kwargs": {"val": "c"}},
        {"name": "local_async", "kwargs": {"val": "local"}},
    ]

    results = await router.execute_concurrently(tool_calls)
    assert results == [
        "result_sync_a",
        "result_async_b",
        "result_sync_c",
        "local_async_local",
    ]


@pytest.mark.asyncio
async def test_router_exception_safety_isolation(local_registry: SkillRegistry) -> None:
    """Tests safety/exception isolation. Failing tools shouldn't crash the entire list."""
    router = ConcurrentToolRouterV5(local_registry)

    def crashing_tool() -> str:
        raise ValueError("Simulated crash")

    local_registry.register_skill("crashing_tool", crashing_tool, "Fails always")

    tool_calls = [
        {"name": "local_async", "kwargs": {"val": "ok1"}},
        {"name": "crashing_tool", "kwargs": {}},
        {"name": "local_async", "kwargs": {"val": "ok2"}},
    ]

    results = await router.execute_concurrently(tool_calls)
    assert len(results) == 3
    assert results[0] == "local_async_ok1"
    assert "Simulated crash" in results[1] or "Tool execution failed" in results[1]
    assert results[2] == "local_async_ok2"


@pytest.mark.asyncio
async def test_router_missing_or_empty_tools(local_registry: SkillRegistry) -> None:
    """Tests non-existent or empty tool name edge cases."""
    router = ConcurrentToolRouterV5(local_registry)

    tool_calls = [
        {"name": "", "kwargs": {}},
        {"name": "missing_tool", "kwargs": {}},
    ]

    results = await router.execute_concurrently(tool_calls)
    assert len(results) == 2
    assert "Empty tool name" in results[0]
    assert "Skill 'missing_tool' not found" in results[1]


@pytest.mark.asyncio
async def test_router_with_mcp_server(local_registry: SkillRegistry) -> None:
    """Tests routing to a registered MCPServer."""
    mcp_registry = SkillRegistry()

    async def mcp_async(val: str) -> str:
        return f"mcp_ok_{val}"

    mcp_registry.register_skill("mcp_async", mcp_async, "MCP dynamic tool")

    mcp_server = MCPServer(MCPExporter(mcp_registry), server_id="")
    servers = {
        "mcp_srv": mcp_server,
    }

    router = ConcurrentToolRouterV5(registry=local_registry, servers=servers)

    tool_calls = [
        {"name": "mcp_srv__mcp_async", "kwargs": {"val": "hello"}},
    ]

    results = await router.execute_concurrently(tool_calls)
    assert results == ["mcp_ok_hello"]
