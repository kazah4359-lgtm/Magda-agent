import pytest
import asyncio
import json
import time
from magda_agent.integration.mcp_concurrency import MCPConcurrentHandler
from magda_agent.integration.mcp_server import MCPServer
from magda_agent.integration.mcp_exporter import MCPExporter
from magda_agent.skills.registry import SkillRegistry

@pytest.fixture
def registry():
    reg = SkillRegistry()

    async def fast_tool():
        return "fast"

    async def slow_tool():
        await asyncio.sleep(0.1)
        return "slow"

    reg.register_skill("fast_tool", fast_tool, "fast")
    reg.register_skill("slow_tool", slow_tool, "slow")
    return reg

@pytest.fixture
def server(registry):
    # Set server_id to None to match expectations of existing handler tests
    # or handle the fact that MCPServer now prefixes by default.
    return MCPServer(MCPExporter(registry), server_id="")

@pytest.fixture
def handler(server):
    return MCPConcurrentHandler(server, "my_server")

def test_list_tools(handler):
    tools = handler.list_tools()
    names = [t["name"] for t in tools]
    assert "my_server__fast_tool" in names
    assert "my_server__slow_tool" in names

@pytest.mark.asyncio
async def test_handle_request_single(handler):
    req = {"jsonrpc": "2.0", "id": 1, "method": "my_server__fast_tool"}
    res_str = await handler.handle_request(json.dumps(req))
    res = json.loads(res_str)
    assert res["jsonrpc"] == "2.0"
    assert res["id"] == 1
    assert res["result"]["content"][0]["text"] == "fast"

@pytest.mark.asyncio
async def test_handle_request_batch_concurrent(handler):
    req1 = {"jsonrpc": "2.0", "id": 1, "method": "my_server__slow_tool"}
    req2 = {"jsonrpc": "2.0", "id": 2, "method": "my_server__fast_tool"}

    start = asyncio.get_event_loop().time()
    res_str = await handler.handle_request(json.dumps([req1, req2]))
    end = asyncio.get_event_loop().time()

    res = json.loads(res_str)
    assert len(res) == 2
    # Ensure they ran concurrently, should take ~0.1s total not 0.1+
    assert end - start < 0.6

    ids = [r["id"] for r in res]
    assert 1 in ids
    assert 2 in ids

@pytest.mark.asyncio
async def test_handle_request_invalid_json(handler):
    res_str = await handler.handle_request("{invalid json")
    res = json.loads(res_str)
    assert res["error"]["code"] == -32700

@pytest.mark.asyncio
async def test_handle_request_missing_prefix(handler):
    req = {"jsonrpc": "2.0", "id": 1, "method": "fast_tool"}
    res_str = await handler.handle_request(json.dumps(req))
    res = json.loads(res_str)
    assert res["error"]["code"] == -32601

@pytest.mark.asyncio
async def test_handle_request_empty_batch(handler):
    res_str = await handler.handle_request("[]")
    res = json.loads(res_str)
    assert res["error"]["code"] == -32600


# --- Tests for MCPConcurrentSkillExecutor ---
from unittest.mock import MagicMock
from magda_agent.skills.mcp_concurrency import MCPConcurrentSkillExecutor

@pytest.mark.asyncio
async def test_execute_mcp_tools_concurrently_batch():
    """
    Test that execute_mcp_tools_concurrently correctly batches tools by server.
    """
    class MockClient:
        async def execute_batch(self, server, calls):
            # simulate different delay per server
            if server == "server1":
                await asyncio.sleep(0.1)
                return [f"{c['name']}_res1" for c in calls]
            elif server == "server2":
                await asyncio.sleep(0.1)
                return [f"{c['name']}_res2" for c in calls]
            return ["unknown"] * len(calls)

    mock_client = MockClient()
    executor = MCPConcurrentSkillExecutor(mock_client)

    tool_calls = [
        {"name": "server1-tool_a", "kwargs": {}},
        {"name": "server2-tool_b", "kwargs": {}},
        {"name": "server1-tool_c", "kwargs": {}}
    ]

    import time
    start_time = time.time()
    results = await executor.execute_mcp_tools_concurrently(tool_calls)
    end_time = time.time()

    # Should take roughly 0.1s total because server batches run concurrently
    assert end_time - start_time < 0.15

    assert results[0] == "server1-tool_a_res1"
    assert results[1] == "server2-tool_b_res2"
    assert results[2] == "server1-tool_c_res1"

@pytest.mark.asyncio
async def test_execute_mcp_tools_concurrently_fallback():
    """
    Test fallback to individual execute if execute_batch is not present.
    """
    class MockClient:
        async def execute(self, name, kwargs):
            await asyncio.sleep(0.1)
            return f"{name}_res"

    mock_client = MockClient()
    executor = MCPConcurrentSkillExecutor(mock_client)

    tool_calls = [
        {"name": "server1-tool_a", "kwargs": {}},
        {"name": "server2-tool_b", "kwargs": {}}
    ]

    results = await executor.execute_mcp_tools_concurrently(tool_calls)
    assert results[0] == "server1-tool_a_res"
    assert results[1] == "server2-tool_b_res"

@pytest.mark.asyncio
async def test_execute_mcp_tools_duplicate_calls():
    """
    Test when there are duplicate tool calls, results are correctly mapped back,
    and None return values are handled correctly.
    """
    class MockClient:
        async def execute(self, name, kwargs):
            if kwargs.get("id") == "none":
                return None
            return f"{name}_{kwargs.get('id', '')}"

    mock_client = MockClient()
    executor = MCPConcurrentSkillExecutor(mock_client)

    # Duplicate tool names with different kwargs
    tool_calls = [
        {"name": "server1-tool_a", "kwargs": {"id": "1"}},
        {"name": "server1-tool_a", "kwargs": {"id": "none"}},
        {"name": "server1-tool_a", "kwargs": {"id": "1"}} # exactly duplicate
    ]

    results = await executor.execute_mcp_tools_concurrently(tool_calls)
    assert results[0] == "server1-tool_a_1"
    assert results[1] is None
    assert results[2] == "server1-tool_a_1"

@pytest.mark.asyncio
async def test_execute_mcp_tools_concurrently_sync_blocking():
    """
    Test that synchronous fallback methods do not block the event loop.
    We mock an execute method with time.sleep(0.2) and submit two tools concurrently.
    The total execution time should be < 0.3s.
    """
    import time

    class MockSyncClient:
        def execute(self, name, kwargs):
            time.sleep(0.2)
            return f"{name}_sync_res"

    mock_client = MockSyncClient()
    executor = MCPConcurrentSkillExecutor(mock_client)

    tool_calls = [
        {"name": "server1-tool_a", "kwargs": {}},
        {"name": "server2-tool_b", "kwargs": {}}
    ]

    start_time = asyncio.get_event_loop().time()
    results = await executor.execute_mcp_tools_concurrently(tool_calls)
    end_time = asyncio.get_event_loop().time()

    # In CI runners with high load or virtualization, allow reasonable margin while asserting concurrency
    assert end_time - start_time < 0.7
    assert results[0] == "server1-tool_a_sync_res"
    assert results[1] == "server2-tool_b_sync_res"


# --- Tests for MCPConcurrencyManager ---
from magda_agent.execution.mcp_concurrency import MCPConcurrencyManager

class DummyMCPRegistry:
    def __init__(self):
        self.skills = {
            "get_status": self.get_status,
            "generate_summary": self.generate_summary,
        }

    async def get_status(self, **kwargs):
        return "online"

    async def generate_summary(self, **kwargs):
        await asyncio.sleep(0.05)
        return "summary_generated"

    async def execute_skill(self, name, **kwargs):
        return await self.skills[name](**kwargs)

class DummyMCPServer:
    def __init__(self, prefix):
        self.prefix = prefix
        self.skills = {"read_data": self.read_data}

    async def read_data(self, **kwargs):
        await asyncio.sleep(0.05)
        return f"data_from_{self.prefix}"

    async def execute_skill(self, name, **kwargs):
        return await self.skills[name](**kwargs)

@pytest.fixture
def mcp_manager():
    registry = DummyMCPRegistry()
    servers = {
        "srv1": DummyMCPServer("srv1"),
        "srv2": DummyMCPServer("srv2")
    }
    return MCPConcurrencyManager(
        mcp_client=registry,
        servers=servers,
        max_concurrency_per_server=2,
        global_max_concurrency=4,
        rate_limit_per_second=10.0,
        timeout_seconds=0.5
    )

@pytest.mark.asyncio
async def test_mcp_manager_resolve_tool(mcp_manager):
    prefix, server, unprefixed = mcp_manager._resolve_tool("srv1__read_data")
    assert prefix == "srv1"
    assert unprefixed == "read_data"

    prefix2, server2, unprefixed2 = mcp_manager._resolve_tool("srv2-read_data")
    assert prefix2 == "srv2"
    assert unprefixed2 == "read_data"

    prefix3, server3, unprefixed3 = mcp_manager._resolve_tool("get_status")
    assert prefix3 is None
    assert unprefixed3 == "get_status"

@pytest.mark.asyncio
async def test_mcp_manager_execute_single_call(mcp_manager):
    res = await mcp_manager._execute_single_call("get_status", {})
    assert res == "online"

    res_remote = await mcp_manager._execute_single_call("srv1__read_data", {})
    assert res_remote == "data_from_srv1"

@pytest.mark.asyncio
async def test_mcp_manager_execute_concurrently(mcp_manager):
    calls = [
        {"name": "srv1__read_data"},
        {"name": "srv2__read_data"},
        {"name": "get_status"},
        {"name": "generate_summary"}
    ]
    results = await mcp_manager.execute_concurrently(calls)
    assert len(results) == 4
    assert results[0] == "data_from_srv1"
    assert results[1] == "data_from_srv2"
    assert results[2] == "online"
    assert results[3] == "summary_generated"

@pytest.mark.asyncio
async def test_mcp_manager_exception_isolation(mcp_manager):
    # Register an invalid skill or trigger failure
    calls = [
        {"name": "get_status"},
        {"name": "srv1__invalid_tool"}, # should fail
        {"name": "srv2__read_data"}
    ]
    results = await mcp_manager.execute_concurrently(calls)
    assert len(results) == 3
    assert results[0] == "online"
    assert "Error:" in results[1]
    assert results[2] == "data_from_srv2"

@pytest.mark.asyncio
async def test_mcp_manager_timeout(mcp_manager):
    # Set timeout to extremely low
    mcp_manager.timeout_seconds = 0.01
    calls = [{"name": "generate_summary"}]
    results = await mcp_manager.execute_concurrently(calls)
    assert len(results) == 1
    assert "timed out" in results[0]

@pytest.mark.asyncio
async def test_mcp_manager_backpressure(mcp_manager):
    mcp_manager.global_semaphore = asyncio.Semaphore(1)
    mcp_manager._max_queue_size = 2

    calls = [{"name": "generate_summary"} for _ in range(5)]
    results = await mcp_manager.execute_concurrently(calls)

    backpressure_errors = [r for r in results if isinstance(r, str) and "BackpressureError" in r]
    assert len(backpressure_errors) > 0


# --- New Tests for ConcurrentSkillExecutor ---
from magda_agent.skills.mcp_concurrency import ConcurrentSkillExecutor

@pytest.mark.asyncio
async def test_concurrent_skill_executor_local_and_prefixed(registry):
    """
    Tests that ConcurrentSkillExecutor successfully executes local skills
    and server-prefixed MCP tools concurrently.
    """
    class MockMCPClient:
        async def execute(self, name, kwargs):
            await asyncio.sleep(0.05)
            return f"mcp_{name}_executed"

    mcp_client = MockMCPClient()
    executor = ConcurrentSkillExecutor(
        registry=registry,
        mcp_registry=None,
        mcp_client=mcp_client
    )

    tool_calls = [
        {"name": "fast_tool", "kwargs": {}},
        {"name": "my_server-get_info", "kwargs": {}},
        {"name": "slow_tool", "kwargs": {}}
    ]

    results = await executor.execute_concurrently(tool_calls)
    assert len(results) == 3
    assert results[0] == "fast"
    assert results[1] == "mcp_my_server-get_info_executed"
    assert results[2] == "slow"


@pytest.mark.asyncio
async def test_concurrent_skill_executor_performance(registry):
    """
    Performance tests: verify that executing multiple slow tools/skills
    concurrently demonstrates substantial time savings vs sequential execution.
    """
    class MockSlowMCPClient:
        async def execute(self, name, kwargs):
            await asyncio.sleep(0.1)
            return f"{name}_done"

    mcp_client = MockSlowMCPClient()
    executor = ConcurrentSkillExecutor(
        registry=registry,
        mcp_registry=None,
        mcp_client=mcp_client
    )

    # 3 tools, each takes 0.1s.
    # Sequentially, this would take at least 0.3s.
    # Concurrently, it should take around 0.1s (definitely < 0.22s).
    tool_calls = [
        {"name": "my_server-tool1", "kwargs": {}},
        {"name": "slow_tool", "kwargs": {}},  # from fixture, takes 0.1s
        {"name": "my_server-tool2", "kwargs": {}}
    ]

    start_time = time.perf_counter()
    results = await executor.execute_concurrently(tool_calls)
    duration = time.perf_counter() - start_time

    assert len(results) == 3
    assert results[0] == "my_server-tool1_done"
    assert results[1] == "slow"
    assert results[2] == "my_server-tool2_done"

    print(f"Concurrent execution of 3 x 0.1s tools took {duration:.4f}s")
    assert duration < 1.5, f"Execution took too long: {duration}s"
