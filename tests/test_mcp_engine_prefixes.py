import pytest
from unittest.mock import MagicMock
import asyncio

from magda_agent.skills.mcp_engine import MCPEngine
from magda_agent.skills.mcp_client import MCPClient
from magda_agent.skills.registry import SkillRegistry


def test_import_mcp_tool_with_server_prefix():
    """
    Test that importing an MCP tool with a server name correctly prefixes
    the tool name and registers the server with the MCP client.
    """
    mock_registry = MagicMock(spec=SkillRegistry)
    mock_mcp_client = MagicMock(spec=MCPClient)

    engine = MCPEngine(registry=mock_registry, mcp_client=mock_mcp_client)

    tool_def = {
        "name": "search_docs",
        "description": "Searches documentation.",
        "inputSchema": {}
    }
    connection_info = {"url": "http://example.com"}

    engine.import_mcp_tool(
        tool_def=tool_def,
        connection_info=connection_info,
        server_name="docs_server"
    )

    # Verify MCP client registered the server, not the tool directly
    mock_mcp_client.register_mcp_server.assert_called_once_with("docs_server", connection_info)
    mock_mcp_client.register_remote_tool.assert_not_called()

    # Verify registry registered the prefixed tool
    mock_registry.register_skill.assert_called_once()
    registered_kwargs = mock_registry.register_skill.call_args.kwargs
    assert registered_kwargs["name"] == "docs_server__search_docs"
    assert registered_kwargs["description"] == "Searches documentation."
    assert registered_kwargs["func"].__name__ == "docs_server__search_docs"
    assert getattr(registered_kwargs["func"], "__mcp_schema__") == {}


def test_import_mcp_tool_without_server_prefix():
    """
    Test that importing an MCP tool without a server name defaults to the original
    behavior of registering the remote tool directly with the MCP client.
    """
    mock_registry = MagicMock(spec=SkillRegistry)
    mock_mcp_client = MagicMock(spec=MCPClient)

    engine = MCPEngine(registry=mock_registry, mcp_client=mock_mcp_client)

    tool_def = {
        "name": "search_web",
        "description": "Searches the web.",
        "inputSchema": {}
    }
    connection_info = {"url": "http://example.com/web"}

    engine.import_mcp_tool(
        tool_def=tool_def,
        connection_info=connection_info
    )

    # Verify MCP client registered the remote tool directly
    mock_mcp_client.register_remote_tool.assert_called_once_with("search_web", connection_info)
    mock_mcp_client.register_mcp_server.assert_not_called()

    # Verify registry registered the tool without prefix
    mock_registry.register_skill.assert_called_once()
    registered_kwargs = mock_registry.register_skill.call_args.kwargs
    assert registered_kwargs["name"] == "search_web"
    assert registered_kwargs["description"] == "Searches the web."
    assert registered_kwargs["func"].__name__ == "search_web"
    assert getattr(registered_kwargs["func"], "__mcp_schema__") == {}


@pytest.mark.asyncio
async def test_imported_prefixed_tool_execution():
    """
    Test that the dynamically wrapped async skill calls the MCP client
    with the correctly prefixed tool name.
    """
    mock_registry = MagicMock(spec=SkillRegistry)
    mock_mcp_client = MagicMock(spec=MCPClient)

    # Make the execute_tool method return a coroutine
    async def mock_execute(*args, **kwargs):
        return "success result"
    mock_mcp_client.execute_tool.side_effect = mock_execute

    engine = MCPEngine(registry=mock_registry, mcp_client=mock_mcp_client)

    tool_def = {
        "name": "get_weather",
        "description": "Gets weather.",
        "inputSchema": {}
    }
    connection_info = {"url": "http://example.com/weather"}

    engine.import_mcp_tool(
        tool_def=tool_def,
        connection_info=connection_info,
        server_name="weather_station"
    )

    registered_func = mock_registry.register_skill.call_args.kwargs["func"]

    # Execute the wrapped skill
    result = await registered_func(location="London")

    assert result == "success result"
    mock_mcp_client.execute_tool.assert_called_once_with("weather_station__get_weather", location="London")
