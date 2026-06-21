import pytest
import asyncio
import json
import threading
from magda_agent.integration.mcp_concurrency_v2 import MCPConcurrentHandlerV2
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
    return MCPConcurrentHandlerV2(server, "my_server", max_concurrency=2)

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

@pytest.mark.asyncio
async def test_semaphore_limits_concurrency(handler):
    # Setup handler with max_concurrency 1 to force sequential execution
    handler.semaphore = asyncio.Semaphore(1)

    req1 = {"jsonrpc": "2.0", "id": 1, "method": "my_server__slow_tool"}
    req2 = {"jsonrpc": "2.0", "id": 2, "method": "my_server__slow_tool"}

    start = asyncio.get_event_loop().time()
    res_str = await handler.handle_request(json.dumps([req1, req2]))
    end = asyncio.get_event_loop().time()

    res = json.loads(res_str)
    assert len(res) == 2

    # Since concurrency is 1, two 0.1s tasks should take at least 0.2s
    assert end - start >= 0.2

@pytest.mark.asyncio
async def test_active_tasks_count(handler):
    # Testing that active_tasks_count goes up and then down

    req = {"jsonrpc": "2.0", "id": 1, "method": "my_server__slow_tool"}

    # Run the request as a background task
    task = asyncio.create_task(handler.handle_request(json.dumps(req)))

    # Give it a moment to start
    await asyncio.sleep(0.01)

    # It should be active now
    assert handler.active_tasks_count == 1

    # Wait for completion
    await task

    # It should be back to 0
    assert handler.active_tasks_count == 0


@pytest.mark.asyncio
async def test_handle_request_exception_in_batch(handler):
    # Test to verify that exceptions during batch processing are caught
    # and returned as valid JSON-RPC error dictionaries rather than crashing

    # We will mock _process_single_request to raise an exception
    original_process = handler._process_single_request

    async def failing_process(req):
        raise ValueError("Simulated failure")

    handler._process_single_request = failing_process

    try:
        req1 = {"jsonrpc": "2.0", "id": 1, "method": "my_server__fast_tool"}
        res_str = await handler.handle_request(json.dumps([req1]))
        res = json.loads(res_str)

        assert len(res) == 1
        assert res[0]["error"]["code"] == -32000
        assert "Simulated failure" in res[0]["error"]["message"]
        assert res[0]["id"] == 1
    finally:
        handler._process_single_request = original_process
