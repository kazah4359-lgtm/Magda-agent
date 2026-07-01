import pytest
from magda_agent.skills.mcp_registry import MCPRegistry
from magda_agent.skills.mcp_dynamic_registry import MCPDynamicRegistrarV4

def test_register_tool_at_runtime_success_sync():
    """Test successful dynamic registration of a synchronous MCP tool via V4."""
    registry = MCPRegistry()
    registrar = MCPDynamicRegistrarV4(registry)

    valid_schema = {
        "name": "test_sync_tool_v4",
        "description": "A tool for testing sync behavior in V4."
    }

    result = registrar.register_tool_at_runtime(valid_schema, is_async=False)

    assert result is True
    assert "test_sync_tool_v4" in registry.list_tools()
    tool = registry.get_tool("test_sync_tool_v4")
    assert tool["name"] == "test_sync_tool_v4"
    assert tool["__is_async__"] is False

def test_register_tool_at_runtime_success_async():
    """Test successful dynamic registration of an asynchronous MCP tool via V4."""
    registry = MCPRegistry()
    registrar = MCPDynamicRegistrarV4(registry)

    valid_schema = {
        "name": "test_async_tool_v4",
        "description": "A tool for testing async behavior in V4."
    }

    result = registrar.register_tool_at_runtime(valid_schema, is_async=True)

    assert result is True
    assert "test_async_tool_v4" in registry.list_tools()
    tool = registry.get_tool("test_async_tool_v4")
    assert tool["name"] == "test_async_tool_v4"
    assert tool["__is_async__"] is True

def test_register_tool_at_runtime_failure():
    """Test failing dynamic registration of an MCP tool due to invalid schema via V4."""
    registry = MCPRegistry()
    registrar = MCPDynamicRegistrarV4(registry)

    invalid_schema = {
        "name": "invalid_tool_v4"
        # Missing description
    }

    result = registrar.register_tool_at_runtime(invalid_schema)

    assert result is False
    assert "invalid_tool_v4" not in registry.list_tools()
