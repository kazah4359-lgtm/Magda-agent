import pytest
import json
import asyncio
from magda_agent.mcp.action_tools_v8 import MCPActionToolManagerV8
from magda_agent.skills.registry import SkillRegistry
from magda_agent.safety.policy import PolicyLayer

@pytest.fixture
def registry():
    reg = SkillRegistry()
    def save_data(data: str):
        return f"Saved: {data}"

    reg.register_skill("save_data", save_data, "Saves some data")

    async def async_action(val: int):
        return f"Async: {val}"

    reg.register_skill("async_action", async_action, "An async action")

    return reg

@pytest.fixture
def policy_layer():
    return PolicyLayer()

@pytest.fixture
def manager(registry, policy_layer):
    mgr = MCPActionToolManagerV8(registry, policy_layer)
    mgr.register_action_tool(
        "save_data",
        "Saves some data",
        {"type": "object", "properties": {"data": {"type": "string"}}, "required": ["data"]}
    )
    mgr.register_action_tool(
        "async_action",
        "An async action",
        {"type": "object", "properties": {"val": {"type": "integer"}}, "required": ["val"]}
    )
    return mgr

def test_register_and_list_tools(manager):
    tools = manager.list_tools()
    assert len(tools) == 2
    names = [t["name"] for t in tools]
    assert "save_data" in names
    assert "async_action" in names

@pytest.mark.asyncio
async def test_handle_list_tools_rpc(manager):
    request = {
        "jsonrpc": "2.0",
        "method": "list_tools",
        "id": "1"
    }
    response_str = await manager.handle_mcp_request(json.dumps(request))
    response = json.loads(response_str)

    assert response["id"] == "1"
    assert "tools" in response["result"]
    assert len(response["result"]["tools"]) == 2

@pytest.mark.asyncio
async def test_call_tool_success(manager):
    request = {
        "jsonrpc": "2.0",
        "method": "call_tool",
        "params": {
            "name": "save_data",
            "arguments": {"data": "important info"}
        },
        "id": "2"
    }
    response_str = await manager.handle_mcp_request(json.dumps(request))
    response = json.loads(response_str)

    assert response["id"] == "2"
    assert response["result"]["content"][0]["text"] == "Saved: important info"
    assert response["result"]["isError"] is False

@pytest.mark.asyncio
async def test_call_async_tool_success(manager):
    request = {
        "jsonrpc": "2.0",
        "method": "call_tool",
        "params": {
            "name": "async_action",
            "arguments": {"val": 42}
        },
        "id": "3"
    }
    response_str = await manager.handle_mcp_request(json.dumps(request))
    response = json.loads(response_str)

    assert response["id"] == "3"
    assert response["result"]["content"][0]["text"] == "Async: 42"

@pytest.mark.asyncio
async def test_policy_gating_denied(manager, registry):
    # Register a tool that policy might deny
    # PolicyLayer denies 'programmer' if it mentions .env
    def programmer(code: str):
        return "Executed"

    registry.register_skill("programmer", programmer, "Executes code")
    manager.register_action_tool("programmer", "Executes code", {})

    request = {
        "jsonrpc": "2.0",
        "method": "call_tool",
        "params": {
            "name": "programmer",
            "arguments": {"code": "read('.env')"}
        },
        "id": "4"
    }
    response_str = await manager.handle_mcp_request(json.dumps(request))
    response = json.loads(response_str)

    assert "error" in response
    assert response["error"]["code"] == -32000
    assert "Policy violation" in response["error"]["message"]

@pytest.mark.asyncio
async def test_invalid_json(manager):
    response_str = await manager.handle_mcp_request("invalid json")
    response = json.loads(response_str)
    assert response["error"]["code"] == -32700

@pytest.mark.asyncio
async def test_missing_method(manager):
    request = {"jsonrpc": "2.0", "id": "5"}
    response_str = await manager.handle_mcp_request(json.dumps(request))
    response = json.loads(response_str)
    assert response["error"]["code"] == -32601

@pytest.mark.asyncio
async def test_tool_not_found(manager):
    request = {
        "jsonrpc": "2.0",
        "method": "call_tool",
        "params": {"name": "ghost_tool"},
        "id": "6"
    }
    response_str = await manager.handle_mcp_request(json.dumps(request))
    response = json.loads(response_str)
    assert response["error"]["code"] == -32601
    assert "not found" in response["error"]["message"]
