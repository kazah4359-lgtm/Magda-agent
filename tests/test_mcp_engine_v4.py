import pytest
from unittest.mock import MagicMock, AsyncMock
from magda_agent.skills.mcp_engine_v4 import MCPEngineV4
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.mcp_client import MCPClient
from magda_agent.architecture.context_hooks import HookRegistry

def test_mcp_engine_v4_import() -> None:
    """Verify MCPEngineV4 reads MCP standard tool definitions and wraps them."""
    registry = SkillRegistry()
    mcp_client = MCPClient()
    mcp_client.register_remote_tool = MagicMock()

    engine = MCPEngineV4(registry, mcp_client)

    tool_def = {
        "name": "external_weather",
        "description": "Get current weather.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
    connection_info = {"url": "http://localhost:8000/mcp"}

    engine.import_mcp_tool(tool_def, connection_info)

    # Verify tool routing is registered
    mcp_client.register_remote_tool.assert_called_once_with("external_weather", connection_info)

    # Verify skill is dynamically registered in Magda
    assert registry.has_skill("external_weather")
    assert registry.descriptions["external_weather"] == "Get current weather."

    # Verify schema preservation
    skill_func = registry.skills["external_weather"]
    assert hasattr(skill_func, "__mcp_schema__")
    assert getattr(skill_func, "__mcp_schema__") == tool_def["inputSchema"]

@pytest.mark.asyncio
async def test_mcp_engine_v4_wrapper_execution_and_hooks() -> None:
    """Verify the dynamically wrapped skill executes correctly and triggers hooks."""
    registry = SkillRegistry()
    mcp_client = MCPClient()
    mcp_client.execute_tool = AsyncMock(return_value="Sunny, 25C")

    engine = MCPEngineV4(registry, mcp_client)
    hook_registry = HookRegistry()
    hook_registry.trigger_broadcast_async = AsyncMock()

    # Bootstrap the plugin
    await engine.bootstrap({"hook_registry": hook_registry})

    tool_def = {"name": "external_weather"}
    connection_info = {"url": "mock"}

    engine.import_mcp_tool(tool_def, connection_info)

    result = await registry.execute_skill("external_weather", location="Paris")

    assert result == "Sunny, 25C"
    mcp_client.execute_tool.assert_awaited_once_with("external_weather", location="Paris")

    # Verify hooks were triggered
    assert hook_registry.trigger_broadcast_async.await_count == 2
    hook_registry.trigger_broadcast_async.assert_any_await("before_tool_use", "external_weather", {"location": "Paris"})
    hook_registry.trigger_broadcast_async.assert_any_await("after_tool_use", "external_weather", "Sunny, 25C")

def test_mcp_engine_v4_invalid_tool_def() -> None:
    """Verify MCPEngineV4 raises ValueError on invalid tool definition without a name."""
    registry = SkillRegistry()
    mcp_client = MCPClient()
    engine = MCPEngineV4(registry, mcp_client)

    with pytest.raises(ValueError, match="must include a 'name'"):
        engine.import_mcp_tool({"description": "No name tool"}, {})

@pytest.mark.asyncio
async def test_mcp_engine_v4_protocol_methods() -> None:
    """Verify the basic ContextPlugin methods."""
    registry = SkillRegistry()
    mcp_client = MCPClient()
    engine = MCPEngineV4(registry, mcp_client)

    assert await engine.ingest("data", {}) == "data"
    assert await engine.assemble(["A", "B"], {}) == "A\nB"
    assert await engine.compact(["A"], {}) == ["A"]

    assert engine.before_retrieval("query", 1) == "query"
    assert engine.after_retrieval(["data"], "query", 1) == ["data"]
    assert engine.before_write("data", 1) == "data"

    # Just asserting no exception on empty pass methods
    engine.after_write("data", 1)
    engine.on_context_update("data", 1)
