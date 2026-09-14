import asyncio
import json
import time
import pytest

from magda_agent.skills.mcp_concurrency_v4 import MCPConcurrencyManagerV4, MCPActionToolConcurrencyV4
from magda_agent.skills.registry import SkillRegistry


@pytest.fixture
def skill_registry_1() -> SkillRegistry:
    reg = SkillRegistry()

    def sync_tool(val: str) -> str:
        time.sleep(0.05)
        return f"sync_res_{val}"

    async def async_tool(val: str) -> str:
        await asyncio.sleep(0.05)
        return f"async_res_{val}"

    reg.register_skill("sync_tool", sync_tool, "Sync skill")
    reg.register_skill("async_tool", async_tool, "Async skill")
    return reg


@pytest.fixture
def skill_registry_2() -> SkillRegistry:
    reg = SkillRegistry()

    def calc_tool(x: int) -> int:
        return x * 2

    reg.register_skill("calc_tool", calc_tool, "Calculator skill")
    return reg


@pytest.mark.asyncio
async def test_mcp_concurrency_v4_execute_concurrently(skill_registry_1: SkillRegistry, skill_registry_2: SkillRegistry) -> None:
    manager = MCPConcurrencyManagerV4(registry=skill_registry_1)
    manager.register_server("srv2", skill_registry_2)

    tool_calls = [
        {"name": "sync_tool", "kwargs": {"val": "hello"}},
        {"name": "async_tool", "kwargs": {"val": "world"}},
        {"name": "srv2__calc_tool", "kwargs": {"x": 21}},
    ]

    start = asyncio.get_event_loop().time()
    results = await manager.execute_concurrently(tool_calls)
    end = asyncio.get_event_loop().time()

    assert results == ["sync_res_hello", "async_res_world", 42]
    # Executing 0.05s sync and 0.05s async concurrently should complete in under 0.2s
    assert end - start < 0.2


@pytest.mark.asyncio
async def test_mcp_concurrency_v4_json_rpc_batch(skill_registry_1: SkillRegistry, skill_registry_2: SkillRegistry) -> None:
    manager = MCPActionToolConcurrencyV4(registry=skill_registry_1)
    manager.register_server("srv2", skill_registry_2)

    payload = [
        {"jsonrpc": "2.0", "id": 1, "method": "sync_tool", "params": {"val": "a"}},
        {"jsonrpc": "2.0", "id": 2, "method": "srv2-calc_tool", "params": {"x": 10}},
    ]

    response_str = await manager.handle_request(json.dumps(payload))
    res = json.loads(response_str)

    assert len(res) == 2
    res_dict = {r["id"]: r["result"]["content"][0]["text"] for r in res}
    assert res_dict[1] == "sync_res_a"
    assert res_dict[2] == "20"


@pytest.mark.asyncio
async def test_mcp_concurrency_v4_error_isolation(skill_registry_1: SkillRegistry) -> None:
    def failing_tool() -> str:
        raise ValueError("Tool execution error")

    skill_registry_1.register_skill("fail_tool", failing_tool, "Failing tool")

    manager = MCPConcurrencyManagerV4(registry=skill_registry_1)

    tool_calls = [
        {"name": "sync_tool", "kwargs": {"val": "ok"}},
        {"name": "fail_tool", "kwargs": {}},
        {"name": "non_existent_tool", "kwargs": {}},
    ]

    results = await manager.execute_concurrently(tool_calls)
    assert results[0] == "sync_res_ok"
    assert "fail_tool" in str(results[1]) or "Error" in str(results[1])
    assert "Error: Skill 'non_existent_tool' not found" in str(results[2])


@pytest.mark.asyncio
async def test_mcp_concurrency_v4_active_tasks_and_list_tools(skill_registry_1: SkillRegistry, skill_registry_2: SkillRegistry) -> None:
    manager = MCPConcurrencyManagerV4(registry=skill_registry_1)
    manager.register_server("remote", skill_registry_2)

    tools = manager.list_tools()
    tool_names = [t.get("name") for t in tools if isinstance(t, dict)]
    assert "sync_tool" in tool_names
    assert "async_tool" in tool_names
    assert "remote__calc_tool" in tool_names

    assert manager.active_tasks_count == 0


@pytest.mark.asyncio
async def test_mcp_concurrency_v4_invalid_payloads(skill_registry_1: SkillRegistry) -> None:
    manager = MCPConcurrencyManagerV4(registry=skill_registry_1)

    res_parse = await manager.handle_request("invalid_json_str{")
    assert "Parse error" in res_parse

    res_empty_batch = await manager.handle_request("[]")
    assert "Invalid Request" in res_empty_batch

    res_missing_method = await manager.handle_request(json.dumps({"jsonrpc": "2.0", "id": 1}))
    assert "Method not found" in res_missing_method
