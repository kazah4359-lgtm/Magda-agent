import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from magda_agent.skills.concurrency import ConcurrentSkillExecutor
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.mcp_engine import MCPEngine
from magda_agent.skills.mcp_client import MCPClient

@pytest.mark.asyncio
async def test_mcp_wrapper_concurrency() -> None:
    """Tests that MCP tools can be executed concurrently without returning unawaited coroutines."""
    registry = SkillRegistry()
    mcp_client = AsyncMock(spec=MCPClient)
    mcp_client.execute_tool.return_value = "mcp tool success"
    mcp_client.register_remote_tool = MagicMock()

    engine = MCPEngine(registry, mcp_client)
    tool_def = {"name": "test_mcp_tool", "description": "test", "inputSchema": {}}
    engine.import_mcp_tool(tool_def, {})

    executor = ConcurrentSkillExecutor(registry)

    # Execute concurrently
    calls = [{"name": "test_mcp_tool", "kwargs": {"param": 1}} for _ in range(3)]
    results = await executor.execute_concurrently(calls)

    assert results == ["mcp tool success", "mcp tool success", "mcp tool success"]
