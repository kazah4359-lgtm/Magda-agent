import pytest
from magda_agent.skills.mcp_registry import MCPRegistry

@pytest.fixture
def registry() -> MCPRegistry:
    return MCPRegistry()

def test_load_valid_tool(registry: MCPRegistry) -> None:
    """Test loading a valid MCP tool schema."""
    valid_schema = {
        "name": "calculator_tool",
        "description": "Performs basic mathematical operations.",
        "inputSchema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    }

    assert registry.load_tool(valid_schema) is True
    assert "calculator_tool" in registry.list_tools()

    tool = registry.get_tool("calculator_tool")
    assert tool["name"] == "calculator_tool"
    assert "inputSchema" in tool

def test_load_invalid_tool_missing_name(registry: MCPRegistry) -> None:
    """Test loading an invalid MCP tool schema (missing name)."""
    invalid_schema = {
        "description": "Missing name field."
    }

    assert registry.load_tool(invalid_schema) is False
    assert len(registry.list_tools()) == 0

def test_load_invalid_tool_missing_description(registry: MCPRegistry) -> None:
    """Test loading an invalid MCP tool schema (missing description)."""
    invalid_schema = {
        "name": "some_tool"
    }

    assert registry.load_tool(invalid_schema) is False
    assert len(registry.list_tools()) == 0

def test_load_invalid_tool_not_dict(registry: MCPRegistry) -> None:
    """Test loading an invalid MCP tool schema (not a dict)."""
    invalid_schema = "this is just a string" # type: ignore

    assert registry.load_tool(invalid_schema) is False
    assert len(registry.list_tools()) == 0

def test_get_nonexistent_tool(registry: MCPRegistry) -> None:
    """Test retrieving a tool that has not been loaded."""
    tool = registry.get_tool("fake_tool")
    assert tool == {}

def test_list_tools(registry: MCPRegistry) -> None:
    """Test listing loaded tools."""
    tool1 = {"name": "tool1", "description": "d1"}
    tool2 = {"name": "tool2", "description": "d2"}

    registry.load_tool(tool1)
    registry.load_tool(tool2)

    tools = registry.list_tools()
    assert len(tools) == 2
    assert "tool1" in tools
    assert "tool2" in tools
