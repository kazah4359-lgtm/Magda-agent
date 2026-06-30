import pytest
from magda_agent.skills.mcp_registry import MCPRegistry
from magda_agent.skills.mcp_dynamic import MCPDynamicRegistrar

def test_register_tool_at_runtime_success():
    """Test successful dynamic registration of an MCP tool."""
    registry = MCPRegistry()
    registrar = MCPDynamicRegistrar(registry)

    valid_schema = {
        "name": "test_tool",
        "description": "A tool for testing."
    }

    result = registrar.register_tool_at_runtime(valid_schema)

    assert result is True
    assert "test_tool" in registry.list_tools()
    assert registry.get_tool("test_tool") == valid_schema

def test_register_tool_at_runtime_failure():
    """Test failing dynamic registration of an MCP tool due to invalid schema."""
    registry = MCPRegistry()
    registrar = MCPDynamicRegistrar(registry)

    invalid_schema = {
        "name": "invalid_tool"
        # Missing description
    }

    result = registrar.register_tool_at_runtime(invalid_schema)

    assert result is False
    assert "invalid_tool" not in registry.list_tools()
