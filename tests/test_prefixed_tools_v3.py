import pytest
from magda_agent.mcp.prefixed_tools_v3 import MCPServerPrefixedToolsV3

def test_format_tool_name():
    manager = MCPServerPrefixedToolsV3()
    assert manager.format_tool_name("server1", "my_tool") == "server1_my_tool"

def test_register_and_get_tool():
    manager = MCPServerPrefixedToolsV3()

    prefixed_name = manager.register_tool(
        server_id="test_server",
        tool_name="search",
        description="Search tool",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}}
    )

    assert prefixed_name == "test_server_search"

    tool = manager.get_tool("test_server_search")
    assert tool is not None
    assert tool["name"] == "test_server_search"
    assert tool["original_name"] == "search"
    assert tool["server_id"] == "test_server"
    assert tool["description"] == "Search tool"
    assert "q" in tool["inputSchema"]["properties"]

def test_get_nonexistent_tool():
    manager = MCPServerPrefixedToolsV3()
    assert manager.get_tool("missing_tool") is None

def test_list_tools():
    manager = MCPServerPrefixedToolsV3()
    manager.register_tool("srv1", "toolA", "descA", {})
    manager.register_tool("srv2", "toolA", "descB", {})

    tools = manager.list_tools()
    assert len(tools) == 2
    names = {t["name"] for t in tools}
    assert "srv1_toolA" in names
    assert "srv2_toolA" in names
