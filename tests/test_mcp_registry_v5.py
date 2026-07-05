import pytest
from unittest.mock import MagicMock
from magda_agent.skills.mcp_registry_v5 import MCPRegistryV5

def test_mcp_registry_v5_init():
    registry = MCPRegistryV5()
    assert registry.mcp_tools == {}

def test_mcp_registry_v5_load_tool_success():
    registry = MCPRegistryV5()
    tool_schema = {"name": "test_tool", "description": "A test tool."}

    # We do not use LLMs directly in this registry, but if we did, we'd mock it here
    mock_llm = MagicMock()

    assert registry.load_tool(tool_schema) is True
    assert registry.mcp_tools["test_tool"] == tool_schema

def test_mcp_registry_v5_load_tool_invalid_schema():
    registry = MCPRegistryV5()

    # Missing description
    invalid_schema = {"name": "test_tool"}
    assert registry.load_tool(invalid_schema) is False
    assert "test_tool" not in registry.mcp_tools

    # Missing name
    invalid_schema2 = {"description": "test"}
    assert registry.load_tool(invalid_schema2) is False

    # Not a dict
    assert registry.load_tool("not a dict") is False # type: ignore

def test_mcp_registry_v5_unload_tool_success():
    registry = MCPRegistryV5()
    tool_schema = {"name": "test_tool", "description": "A test tool."}
    registry.load_tool(tool_schema)

    assert registry.unload_tool("test_tool") is True
    assert "test_tool" not in registry.mcp_tools

def test_mcp_registry_v5_unload_tool_not_found():
    registry = MCPRegistryV5()
    assert registry.unload_tool("nonexistent_tool") is False
