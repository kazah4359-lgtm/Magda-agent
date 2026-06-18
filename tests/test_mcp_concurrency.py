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
