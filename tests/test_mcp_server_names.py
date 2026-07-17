import pytest
from magda_agent.integration.mcp_server_names import MCPToolNameParser

def test_parse_with_prefix():
    """Test parsing a tool name with a valid server prefix."""
    prefix, name = MCPToolNameParser.parse("weather_get_forecast")
    assert prefix == "weather"
    assert name == "get_forecast"

def test_parse_without_prefix():
    """Test parsing a tool name without an underscore (no prefix)."""
    prefix, name = MCPToolNameParser.parse("getforecast")
    assert prefix is None
    assert name == "getforecast"

def test_parse_empty_string():
    """Test parsing an empty string."""
    prefix, name = MCPToolNameParser.parse("")
    assert prefix is None
    assert name == ""

def test_parse_leading_underscore():
    """Test parsing a string with a leading underscore (invalid prefix)."""
    prefix, name = MCPToolNameParser.parse("_get_forecast")
    assert prefix is None
    assert name == "_get_forecast"

def test_parse_trailing_underscore():
    """Test parsing a string with a trailing underscore (invalid name)."""
    prefix, name = MCPToolNameParser.parse("weather_")
    assert prefix is None
    assert name == "weather_"

def test_apply_prefix_normal():
    """Test applying a prefix to a normal tool name."""
    prefixed = MCPToolNameParser.apply_prefix("weather", "get_forecast")
    assert prefixed == "weather_get_forecast"

def test_apply_prefix_already_prefixed():
    """Test applying a prefix to a tool name that already has it."""
    prefixed = MCPToolNameParser.apply_prefix("weather", "weather_get_forecast")
    assert prefixed == "weather_get_forecast"

def test_apply_prefix_empty_server_id():
    """Test applying an empty prefix."""
    prefixed = MCPToolNameParser.apply_prefix("", "get_forecast")
    assert prefixed == "get_forecast"

def test_apply_prefix_different_prefix():
    """Test applying a prefix to a tool name that has a different prefix."""
    prefixed = MCPToolNameParser.apply_prefix("weather", "math_add")
    assert prefixed == "weather_math_add"
