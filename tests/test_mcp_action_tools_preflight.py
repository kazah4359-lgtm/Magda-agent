import json
import pytest
from unittest.mock import MagicMock
from magda_agent.mcp.action_tools_v8 import MCPActionToolManagerV8
from magda_agent.skills.registry import SkillRegistry
from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.taint import mark_tainted

@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=SkillRegistry)
    registry.has_skill.side_effect = lambda name: name in ["get_weather", "unsafe_execute"]
    return registry

@pytest.fixture
def mock_policy():
    policy = MagicMock(spec=PolicyLayer)
    policy.evaluate.return_value = (True, "Allowed")
    return policy

@pytest.fixture
def manager(mock_registry, mock_policy):
    manager = MCPActionToolManagerV8(registry=mock_registry, policy_layer=mock_policy)
    manager.register_action_tool("get_weather", "Gets weather", {})
    return manager

@pytest.mark.asyncio
async def test_call_tool_preflight_blocks_hazardous_shell(manager):
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "call_tool",
        "params": {
            "name": "get_weather",
            "arguments": {"location": "San Francisco; rm -rf /"}
        },
        "id": 1
    })
    response_str = await manager.handle_mcp_request(payload)
    response = json.loads(response_str)
    assert "error" in response
    assert response["error"]["code"] == -32000
    assert "Hazardous shell pattern detected" in response["error"]["message"]

@pytest.mark.asyncio
async def test_call_tool_preflight_blocks_path_traversal(manager):
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "call_tool",
        "params": {
            "name": "get_weather",
            "arguments": {"location": "../../../etc/passwd"}
        },
        "id": 1
    })
    response_str = await manager.handle_mcp_request(payload)
    response = json.loads(response_str)
    assert "error" in response
    assert response["error"]["code"] == -32000
    assert "Hazardous path traversal pattern detected" in response["error"]["message"]

@pytest.mark.asyncio
async def test_call_tool_preflight_blocks_tainted_data(manager, monkeypatch):
    request_with_taint = {
        "jsonrpc": "2.0",
        "method": "call_tool",
        "params": {
            "name": "get_weather",
            "arguments": {"location": mark_tainted("San Francisco")}
        },
        "id": 1
    }

    # Simulate receiving a payload that would result in a tainted dictionary after json.loads
    # We bypass json.loads in manager to directly pass tainted data
    async def mock_handle_mcp_request_direct(payload_dict):
        req_id = payload_dict.get("id")
        method = payload_dict.get("method")
        params = payload_dict.get("params", {})

        if method == "call_tool":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            virtual_request = {
                "jsonrpc": "2.0",
                "method": tool_name,
                "params": arguments
            }
            is_valid, err_code, err_msg = manager.validator.validate_request_dict(virtual_request)
            if not is_valid:
                return manager._error_response(req_id, err_code, err_msg)
        return await manager.handle_mcp_request(json.dumps(payload_dict))

    response_str = await mock_handle_mcp_request_direct(request_with_taint)
    response = json.loads(response_str)
    assert "error" in response
    assert response["error"]["code"] == -32000
    assert "Tainted data detected" in response["error"]["message"]

@pytest.mark.asyncio
async def test_call_tool_preflight_blocks_forbidden_tool(manager):
    # 'unsafe_execute' is in the default forbidden list of MCPPreflightValidator
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "call_tool",
        "params": {
            "name": "unsafe_execute",
            "arguments": {"command": "ls"}
        },
        "id": 1
    })
    # We need to register it first in manager to reach the call_tool logic
    manager.register_action_tool("unsafe_execute", "Unsafe execution", {})

    response_str = await manager.handle_mcp_request(payload)
    response = json.loads(response_str)
    assert "error" in response
    assert response["error"]["code"] == -32000
    assert "blacklisted" in response["error"]["message"]

@pytest.mark.asyncio
async def test_call_tool_preflight_passes_safe_request(manager):
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "call_tool",
        "params": {
            "name": "get_weather",
            "arguments": {"location": "San Francisco"}
        },
        "id": 1
    })
    response_str = await manager.handle_mcp_request(payload)
    response = json.loads(response_str)
    assert "result" in response
    assert response["result"]["isError"] is False
