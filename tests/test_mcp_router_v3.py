import pytest
from magda_agent.skills.mcp_router_v3 import MCPRouterV3

def test_register_server():
    router = MCPRouterV3()
    def mock_handler(tool, args): pass

    router.register_server("test_server", mock_handler)
    assert "test_server" in router.server_handlers

def test_route_tool_success():
    router = MCPRouterV3()

    def weather_handler(tool_name, args):
        if tool_name == "get_forecast":
            return f"Weather for {args.get('location')} is sunny."
        return "Unknown weather tool"

    router.register_server("weather", weather_handler)

    result = router.route_tool("weather_get_forecast", {"location": "London"})
    assert result == "Weather for London is sunny."

def test_route_tool_longest_prefix():
    router = MCPRouterV3()

    router.register_server("db", lambda t, a: f"db: {t}")
    router.register_server("db_admin", lambda t, a: f"db_admin: {t}")

    # Should match db_admin, not db
    res1 = router.route_tool("db_admin_drop_table", {})
    assert res1 == "db_admin: drop_table"

    res2 = router.route_tool("db_query", {})
    assert res2 == "db: query"

def test_route_tool_missing_prefix():
    router = MCPRouterV3()

    with pytest.raises(ValueError, match="must be prefixed with a server name"):
        router.route_tool("noprefix", {})

def test_route_tool_unknown_server():
    router = MCPRouterV3()
    router.register_server("weather", lambda t, a: None)

    with pytest.raises(ValueError, match="No registered MCP server handler found"):
        router.route_tool("unknown_get_data", {})
