import pytest
import asyncio
import json
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
    return MCPServer(MCPExporter(registry))

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
    assert end - start < 0.15

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
