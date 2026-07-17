import asyncio
import json
import threading
import time
import pytest

from magda_agent.integration.mcp_concurrency_v3 import MCPConcurrentHandlerV3
from magda_agent.integration.mcp_server import MCPServer
from magda_agent.integration.mcp_exporter import MCPExporter
from magda_agent.skills.registry import SkillRegistry


@pytest.fixture
def registry_1() -> SkillRegistry:
    """Provides a SkillRegistry with sync and async skills."""
    reg = SkillRegistry()

    def sync_tool(arg: str) -> str:
        time.sleep(0.1)
        return f"sync_1_result_{arg}"

    async def async_tool(arg: str) -> str:
        await asyncio.sleep(0.1)
        return f"async_1_result_{arg}"

    reg.register_skill("sync_tool", sync_tool, "A sync skill")
    reg.register_skill("async_tool", async_tool, "An async skill")
    return reg


@pytest.fixture
def registry_2() -> SkillRegistry:
    """Provides another SkillRegistry with different skills."""
    reg = SkillRegistry()

    def other_tool(arg: str) -> str:
        return f"sync_2_result_{arg}"

    reg.register_skill("other_tool", other_tool, "Other sync skill")
    return reg


@pytest.fixture
def server_1(registry_1: SkillRegistry) -> MCPServer:
    """Initializes MCPServer with registry_1."""
    return MCPServer(MCPExporter(registry_1), server_id="")


@pytest.fixture
def server_2(registry_2: SkillRegistry) -> MCPServer:
    """Initializes MCPServer with registry_2."""
    return MCPServer(MCPExporter(registry_2), server_id="")


@pytest.fixture
def handler_v3(server_1: MCPServer, server_2: MCPServer) -> MCPConcurrentHandlerV3:
    """Provides an MCPConcurrentHandlerV3 with two registered servers."""
    handler = MCPConcurrentHandlerV3(server_1, "server_one", max_concurrency=5)
    handler.register_server("server_two", server_2)
    return handler


def test_list_tools_v3(handler_v3: MCPConcurrentHandlerV3) -> None:
    """Verifies that all tools across servers are listed with correct prefixes."""
    tools = handler_v3.list_tools()
    names = [t["name"] for t in tools]

    assert "server_one__sync_tool" in names
    assert "server_one__async_tool" in names
    assert "server_two__other_tool" in names


@pytest.mark.asyncio
async def test_handle_request_single_v3(handler_v3: MCPConcurrentHandlerV3) -> None:
    """Tests executing a single request via standard prefix."""
    req = {"jsonrpc": "2.0", "id": "req-1", "method": "server_one__async_tool", "params": {"arg": "abc"}}
    res_str = await handler_v3.handle_request(json.dumps(req))
    res = json.loads(res_str)

    assert res["jsonrpc"] == "2.0"
    assert res["id"] == "req-1"
    assert res["result"]["content"][0]["text"] == "async_1_result_abc"


@pytest.mark.asyncio
async def test_handle_request_custom_separator(handler_v3: MCPConcurrentHandlerV3) -> None:
    """Tests executing requests with custom separators like dash or underscore."""
    req_dash = {"jsonrpc": "2.0", "id": "req-dash", "method": "server_one-sync_tool", "params": {"arg": "dash"}}
    res_str = await handler_v3.handle_request(json.dumps(req_dash))
    res = json.loads(res_str)
    assert res["result"]["content"][0]["text"] == "sync_1_result_dash"

    req_underscore = {"jsonrpc": "2.0", "id": "req-under", "method": "server_two_other_tool", "params": {"arg": "under"}}
    res_str_under = await handler_v3.handle_request(json.dumps(req_underscore))
    res_under = json.loads(res_str_under)
    assert res_under["result"]["content"][0]["text"] == "sync_2_result_under"


@pytest.mark.asyncio
async def test_handle_request_batch_concurrent_v3(handler_v3: MCPConcurrentHandlerV3) -> None:
    """Tests parallel execution of sync and async tools to verify concurrency."""
    req1 = {"jsonrpc": "2.0", "id": 1, "method": "server_one__sync_tool", "params": {"arg": "x"}}
    req2 = {"jsonrpc": "2.0", "id": 2, "method": "server_one__async_tool", "params": {"arg": "y"}}
    req3 = {"jsonrpc": "2.0", "id": 3, "method": "server_two__other_tool", "params": {"arg": "z"}}

    start = asyncio.get_event_loop().time()
    res_str = await handler_v3.handle_request(json.dumps([req1, req2, req3]))
    end = asyncio.get_event_loop().time()

    res = json.loads(res_str)
    assert len(res) == 3

    # Both req1 (sync sleep 0.1s) and req2 (async sleep 0.1s) run concurrently.
    # Total execution should be around ~0.1s, much less than sequential 0.2s.
    assert end - start < 0.28

    # Verify all responses are correct
    results = {r["id"]: r["result"]["content"][0]["text"] for r in res}
    assert results[1] == "sync_1_result_x"
    assert results[2] == "async_1_result_y"
    assert results[3] == "sync_2_result_z"


@pytest.mark.asyncio
async def test_semaphore_limit_v3(handler_v3: MCPConcurrentHandlerV3) -> None:
    """Verifies that the semaphore limits parallel concurrency as configured."""
    handler_v3.semaphore = asyncio.Semaphore(1)  # Force sequential execution

    req1 = {"jsonrpc": "2.0", "id": 1, "method": "server_one__async_tool", "params": {"arg": "a"}}
    req2 = {"jsonrpc": "2.0", "id": 2, "method": "server_one__async_tool", "params": {"arg": "b"}}

    start = asyncio.get_event_loop().time()
    res_str = await handler_v3.handle_request(json.dumps([req1, req2]))
    end = asyncio.get_event_loop().time()

    res = json.loads(res_str)
    assert len(res) == 2

    # Since concurrency is limited to 1, both 0.1s tasks run sequentially, taking >= 0.2s
    assert end - start >= 0.18


@pytest.mark.asyncio
async def test_active_tasks_count_v3(handler_v3: MCPConcurrentHandlerV3) -> None:
    """Verifies thread-safe tracking of the active_tasks_count."""
    req = {"jsonrpc": "2.0", "id": 1, "method": "server_one__async_tool", "params": {"arg": "active"}}

    # Verify starting active count is 0
    assert handler_v3.active_tasks_count == 0

    # Start task in background
    task = asyncio.create_task(handler_v3.handle_request(json.dumps(req)))

    # Give the task a moment to start and grab the semaphore
    await asyncio.sleep(0.01)

    # Active count should now be 1
    assert handler_v3.active_tasks_count == 1

    await task

    # Completed task should result in active count returning to 0
    assert handler_v3.active_tasks_count == 0


@pytest.mark.asyncio
async def test_error_handling_v3(handler_v3: MCPConcurrentHandlerV3) -> None:
    """Tests parsing errors, missing prefixes, invalid methods, and exceptions."""
    # Invalid JSON
    res_str = await handler_v3.handle_request("{invalid_json}")
    res = json.loads(res_str)
    assert res["error"]["code"] == -32700

    # Empty batch
    res_str = await handler_v3.handle_request("[]")
    res = json.loads(res_str)
    assert res["error"]["code"] == -32600

    # Missing prefix (method name that doesn't map to any server prefix and no default server "")
    req_no_prefix = {"jsonrpc": "2.0", "id": 1, "method": "unknown_tool", "params": {}}
    res_str = await handler_v3.handle_request(json.dumps(req_no_prefix))
    res = json.loads(res_str)
    assert res["error"]["code"] == -32601
    assert "Method not found" in res["error"]["message"]

    # Exception inside tool execution gets caught gracefully
    def failing_tool() -> str:
        raise RuntimeError("Something went wrong")

    handler_v3.servers["server_one"].exporter.registry.register_skill(
        "fail_tool", failing_tool, "Failing skill"
    )

    req_fail = {"jsonrpc": "2.0", "id": 9, "method": "server_one__fail_tool", "params": {}}
    res_str = await handler_v3.handle_request(json.dumps(req_fail))
    res = json.loads(res_str)
    assert res["error"]["code"] == -32000
    assert "Something went wrong" in res["error"]["message"]
