"""Unit tests for MCPTaintContextExporterV1."""
import pytest
import asyncio
from magda_agent.skills.registry import SkillRegistry
from magda_agent.safety.taint import mark_tainted
from magda_agent.integration.mcp_taint_context_v1 import MCPTaintContextExporterV1


def sample_echo_skill(text: str) -> str:
    """Echoes input text."""
    return f"Echo: {text}"


async def sample_async_skill(text: str) -> str:
    """Async echoes input text."""
    await asyncio.sleep(0.01)
    return f"Async Echo: {text}"


def sample_failing_skill(val: str) -> str:
    """Skill that raises error."""
    raise ValueError("Simulated skill failure")


def test_list_tools():
    """Test tool listing with taint isolation flag."""
    registry = SkillRegistry()
    registry.register_skill("echo", sample_echo_skill, description="Echo skill")

    exporter = MCPTaintContextExporterV1(registry)
    tools = exporter.list_tools()

    assert len(tools) == 1
    tool = tools[0]
    assert tool["name"] == "echo"
    assert tool["description"] == "Echo skill"
    assert tool["taintIsolation"] is True
    assert "inputSchema" in tool


def test_call_tool_untainted():
    """Test calling a tool with untainted inputs."""
    registry = SkillRegistry()
    registry.register_skill("echo", sample_echo_skill, description="Echo skill")

    exporter = MCPTaintContextExporterV1(registry)
    response = exporter.call_tool("echo", {"text": "hello"})

    assert response["isError"] is False
    assert response["tainted"] is False
    assert response["content"][0]["text"] == "Echo: hello"


def test_call_tool_tainted_input():
    """Test calling a tool with tainted inputs propagates taint tag."""
    registry = SkillRegistry()
    registry.register_skill("echo", sample_echo_skill, description="Echo skill")

    exporter = MCPTaintContextExporterV1(registry)
    tainted_args = mark_tainted({"text": "secret_data"})

    response = exporter.call_tool("echo", tainted_args)

    assert response["isError"] is False
    assert response["tainted"] is True
    assert response["content"][0]["text"] == "Echo: secret_data"


@pytest.mark.asyncio
async def test_call_tool_async_untainted():
    """Test calling an async tool with untainted inputs."""
    registry = SkillRegistry()
    registry.register_skill("async_echo", sample_async_skill, description="Async echo")

    exporter = MCPTaintContextExporterV1(registry)
    response = await exporter.call_tool_async("async_echo", {"text": "async_hello"})

    assert response["isError"] is False
    assert response["tainted"] is False
    assert response["content"][0]["text"] == "Async Echo: async_hello"


@pytest.mark.asyncio
async def test_call_tool_async_tainted_input():
    """Test calling an async tool with tainted inputs propagates taint tag."""
    registry = SkillRegistry()
    registry.register_skill("async_echo", sample_async_skill, description="Async echo")

    exporter = MCPTaintContextExporterV1(registry)
    tainted_args = mark_tainted({"text": "async_secret"})

    response = await exporter.call_tool_async("async_echo", tainted_args)

    assert response["isError"] is False
    assert response["tainted"] is True
    assert response["content"][0]["text"] == "Async Echo: async_secret"


def test_call_tool_not_found():
    """Test error response when calling a non-existent tool."""
    registry = SkillRegistry()
    exporter = MCPTaintContextExporterV1(registry)

    response = exporter.call_tool("unknown_tool", {})
    assert response["isError"] is True
    assert response["tainted"] is False
    assert "not found" in response["content"][0]["text"]


def test_call_tool_failure():
    """Test handling of tool execution exception."""
    registry = SkillRegistry()
    registry.register_skill("failing", sample_failing_skill, description="Failing skill")

    exporter = MCPTaintContextExporterV1(registry)
    response = exporter.call_tool("failing", {"val": "test"})

    assert response["isError"] is True
    assert response["tainted"] is False
    assert "Error executing tool" in response["content"][0]["text"] or "Simulated skill failure" in response["content"][0]["text"]
