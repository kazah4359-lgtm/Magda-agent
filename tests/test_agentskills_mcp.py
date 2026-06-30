import pytest
from magda_agent.integration.agentskills_mcp import AgentSkillsMCPConverter

def test_convert_to_mcp_tool_with_parameters():
    """Test converting a skill with 'parameters' mapping to 'inputSchema'."""
    agentskills_tool = {
        "name": "test_skill",
        "description": "A test skill.",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"type": "string"}
            },
            "required": ["arg1"]
        }
    }

    mcp_tool = AgentSkillsMCPConverter.convert_to_mcp_tool(agentskills_tool)

    assert mcp_tool["name"] == "test_skill"
    assert mcp_tool["description"] == "A test skill."
    assert "inputSchema" in mcp_tool
    assert "parameters" not in mcp_tool
    assert mcp_tool["inputSchema"] == agentskills_tool["parameters"]

def test_convert_to_mcp_tool_with_inputschema_fallback():
    """Test converting a skill that already uses 'inputSchema'."""
    tool = {
        "name": "test_skill",
        "description": "A test skill.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "arg1": {"type": "string"}
            },
            "required": ["arg1"]
        }
    }

    mcp_tool = AgentSkillsMCPConverter.convert_to_mcp_tool(tool)

    assert mcp_tool["name"] == "test_skill"
    assert mcp_tool["inputSchema"] == tool["inputSchema"]

def test_convert_to_mcp_tool_missing_schema():
    """Test converting a skill missing schema definition."""
    tool = {
        "name": "test_skill",
        "description": "A test skill."
    }

    mcp_tool = AgentSkillsMCPConverter.convert_to_mcp_tool(tool)

    assert mcp_tool["name"] == "test_skill"
    assert "inputSchema" in mcp_tool
    assert mcp_tool["inputSchema"] == {
        "type": "object",
        "properties": {},
        "required": []
    }

def test_convert_all():
    """Test converting a list of skills."""
    agentskills_tools = [
        {
            "name": "tool1",
            "description": "Tool 1",
            "parameters": {"type": "object", "properties": {}}
        },
        {
            "name": "tool2",
            "description": "Tool 2",
            "parameters": {"type": "object", "properties": {}}
        }
    ]

    mcp_tools = AgentSkillsMCPConverter.convert_all(agentskills_tools)

    assert len(mcp_tools) == 2
    assert mcp_tools[0]["name"] == "tool1"
    assert mcp_tools[0]["inputSchema"] == agentskills_tools[0]["parameters"]
    assert mcp_tools[1]["name"] == "tool2"
    assert mcp_tools[1]["inputSchema"] == agentskills_tools[1]["parameters"]

def test_create_jsonrpc_request_with_req_id():
    """Test creating a JSON-RPC request with a specified ID."""
    req = AgentSkillsMCPConverter.create_jsonrpc_request("my_method", {"arg": 1}, req_id="test-id")

    assert req["jsonrpc"] == "2.0"
    assert req["id"] == "test-id"
    assert req["method"] == "my_method"
    assert req["params"] == {"arg": 1}

def test_create_jsonrpc_request_without_req_id():
    """Test creating a JSON-RPC request without a specified ID generates a UUID."""
    req = AgentSkillsMCPConverter.create_jsonrpc_request("my_method", {"arg": 1})

    assert req["jsonrpc"] == "2.0"
    assert "id" in req
    assert isinstance(req["id"], str)
    assert req["method"] == "my_method"
    assert req["params"] == {"arg": 1}
