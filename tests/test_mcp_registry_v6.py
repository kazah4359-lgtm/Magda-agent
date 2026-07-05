import pytest
from unittest.mock import MagicMock
from magda_agent.skills.mcp_registry_v6 import MCPRegistryV6, MCPAdapter

def test_mcp_registry_v6_init():
    registry = MCPRegistryV6()
    assert registry.mcp_tools == {}
    assert registry.adapters == []

def test_mcp_registry_v6_load_tool_success():
    registry = MCPRegistryV6()
    tool_schema = {"name": "test_tool", "description": "A test tool.", "inputSchema": {"type": "object"}}

    assert registry.load_tool(tool_schema) is True
    assert registry.mcp_tools["test_tool"] == tool_schema

def test_mcp_registry_v6_load_tool_invalid_schema():
    registry = MCPRegistryV6()

    # Missing description
    invalid_schema = {"name": "test_tool"}
    assert registry.load_tool(invalid_schema) is False
    assert "test_tool" not in registry.mcp_tools

    # Missing name
    invalid_schema2 = {"description": "test"}
    assert registry.load_tool(invalid_schema2) is False

    # Not a dict
    assert registry.load_tool("not a dict") is False # type: ignore

    # Invalid inputSchema type
    invalid_schema3 = {"name": "test_tool", "description": "test", "inputSchema": "not_a_dict"}
    assert registry.load_tool(invalid_schema3) is False

def test_mcp_registry_v6_unload_tool_success():
    registry = MCPRegistryV6()
    tool_schema = {"name": "test_tool", "description": "A test tool."}
    registry.load_tool(tool_schema)

    assert registry.unload_tool("test_tool") is True
    assert "test_tool" not in registry.mcp_tools

def test_mcp_registry_v6_unload_tool_not_found():
    registry = MCPRegistryV6()
    assert registry.unload_tool("nonexistent_tool") is False

def test_mcp_registry_v6_get_tool():
    registry = MCPRegistryV6()
    tool_schema = {"name": "test_tool", "description": "A test tool."}
    registry.load_tool(tool_schema)

    assert registry.get_tool("test_tool") == tool_schema
    assert registry.get_tool("not_found") == {}

def test_mcp_registry_v6_list_tools():
    registry = MCPRegistryV6()
    registry.load_tool({"name": "tool1", "description": "First tool"})
    registry.load_tool({"name": "tool2", "description": "Second tool"})

    tools = registry.list_tools()
    assert len(tools) == 2
    assert "tool1" in tools
    assert "tool2" in tools

def test_mcp_registry_v6_sync_from_adapters():
    registry = MCPRegistryV6()

    mock_adapter = MagicMock(spec=MCPAdapter)
    mock_adapter.get_tools.return_value = [
        {"name": "dynamic_tool_1", "description": "Dynamic 1"},
        {"name": "dynamic_tool_2", "description": "Dynamic 2"},
        {"name": "invalid_dynamic"} # Will fail validation
    ]

    registry.register_adapter(mock_adapter)

    loaded_count = registry.sync_from_adapters()

    assert loaded_count == 2
    assert "dynamic_tool_1" in registry.mcp_tools
    assert "dynamic_tool_2" in registry.mcp_tools
    assert "invalid_dynamic" not in registry.mcp_tools
    mock_adapter.get_tools.assert_called_once()
