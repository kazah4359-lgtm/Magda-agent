import pytest
from unittest.mock import MagicMock
from magda_agent.integration.mcp_export import MCPExporter

@pytest.fixture
def mock_registry():
    registry = MagicMock()
    # Provide a simple skill mock
    def test_skill(arg1: str, arg2: int = 0):
        return f"result_{arg1}_{arg2}"

    registry.skills = {"test_skill": test_skill}
    registry.descriptions = {"test_skill": "A test skill."}

    def execute_skill(name, **kwargs):
        if name == "test_skill":
            return test_skill(**kwargs)
        raise ValueError("Skill not found")

    registry.execute_skill = execute_skill
    registry.has_skill = lambda name: name == "test_skill"
    return registry

def test_list_tools(mock_registry):
    exporter = MCPExporter(mock_registry)
    tools = exporter.list_tools()

    assert len(tools) == 1
    tool = tools[0]
    assert tool["name"] == "test_skill"
    assert tool["description"] == "A test skill."
    assert "inputSchema" in tool
    schema = tool["inputSchema"]
    assert schema["type"] == "object"
    assert "arg1" in schema["properties"]
    assert "arg2" in schema["properties"]
    assert schema["properties"]["arg1"]["type"] == "string"
    assert schema["properties"]["arg2"]["type"] == "integer"
    assert "arg1" in schema["required"]
    assert "arg2" not in schema["required"]

def test_call_tool(mock_registry):
    exporter = MCPExporter(mock_registry)
    result = exporter.call_tool("test_skill", {"arg1": "hello", "arg2": 42})
    assert result["isError"] is False
    assert result["content"][0]["text"] == "result_hello_42"

def test_call_tool_not_found(mock_registry):
    exporter = MCPExporter(mock_registry)
    result = exporter.call_tool("unknown_skill", {})
    assert result["isError"] is True
    assert "Error" in result["content"][0]["text"]

@pytest.mark.asyncio
async def test_call_tool_async(mock_registry):
    # Make a mock async skill
    async def async_test_skill(arg: str):
        return f"async_{arg}"

    mock_registry.skills["async_test_skill"] = async_test_skill
    mock_registry.descriptions["async_test_skill"] = "Async test."

    def execute_skill(name, **kwargs):
        if name == "test_skill":
            return mock_registry.skills["test_skill"](**kwargs)
        if name == "async_test_skill":
            return async_test_skill(**kwargs)
        raise ValueError("Skill not found")

    mock_registry.execute_skill = execute_skill
    mock_registry.has_skill = lambda name: name in ["test_skill", "async_test_skill"]

    exporter = MCPExporter(mock_registry)
    result = await exporter.call_tool_async("async_test_skill", {"arg": "world"})
    assert result["isError"] is False
    assert result["content"][0]["text"] == "async_world"

@pytest.mark.asyncio
async def test_call_tool_async_error(mock_registry):
    exporter = MCPExporter(mock_registry)
    result = await exporter.call_tool_async("unknown_skill", {})
    assert result["isError"] is True
    assert "Error" in result["content"][0]["text"]
