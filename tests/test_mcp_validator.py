import pytest
import jsonschema
from typing import Dict, Any
from magda_agent.skills.mcp_validator import MCPActionToolValidator, validate_mcp_tool

def test_validate_schema_valid() -> None:
    schema = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": {
            "type": "object",
            "properties": {
                "arg1": {"type": "string"}
            }
        }
    }
    # Should not raise any exception
    MCPActionToolValidator.validate_schema(schema)

def test_validate_schema_missing_name() -> None:
    schema = {
        "description": "A test tool"
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        MCPActionToolValidator.validate_schema(schema)

def test_validate_schema_invalid_type() -> None:
    schema = {
        "name": 123,  # Invalid type, should be string
        "description": "A test tool"
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        MCPActionToolValidator.validate_schema(schema)

def test_validate_schema_invalid_input_schema_type() -> None:
    schema = {
        "name": "test_tool",
        "description": "A test tool",
        "inputSchema": {
            "type": "array" # Invalid type, should be object
        }
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        MCPActionToolValidator.validate_schema(schema)

def test_validate_mcp_tool_decorator_valid() -> None:
    class MockRegistry:
        @validate_mcp_tool
        def register_tool(self, tool_schema: Dict[str, Any]) -> bool:
            return True

    registry = MockRegistry()
    schema = {
        "name": "test_tool",
        "description": "A test tool"
    }
    assert registry.register_tool(schema) is True

def test_validate_mcp_tool_decorator_invalid() -> None:
    class MockRegistry:
        @validate_mcp_tool
        def register_tool(self, tool_schema: Dict[str, Any]) -> bool:
            return True

    registry = MockRegistry()
    schema = {
        "description": "A test tool" # Missing name
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        registry.register_tool(tool_schema=schema)

def test_validate_mcp_tool_decorator_positional_invalid() -> None:
    class MockRegistry:
        @validate_mcp_tool
        def register_tool(self, tool_schema: Dict[str, Any]) -> bool:
            return True

    registry = MockRegistry()
    schema = {
        "description": "A test tool" # Missing name
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        registry.register_tool(schema) # Call with positional argument
