import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from magda_agent.skills.mcp_client import MCPClient
from magda_agent.agents.generator_agent import GeneratorAgent
from magda_agent.planning.planner import Planner
from magda_agent.skills.registry import SkillRegistry
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_mcp_client_registration_and_execution():
    client = MCPClient(timeout=1.0)
    client.register_remote_tool("test_mcp.search", {"url": "mock"})

    assert client.has_tool("test_mcp.search") is True
    assert client.has_tool("local.skill") is False

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"jsonrpc": "2.0", "result": "Mocked MCP Response", "id": "123"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = await client.execute_tool("test_mcp.search", q="magda")
        assert result == "Mocked MCP Response"
        mock_post.assert_called_once()

    error_result = await client.execute_tool("unknown.tool")
    assert "Error: Remote MCP skill" in error_result

@pytest.mark.asyncio
async def test_generator_agent_with_mcp_client():
    mock_llm = MagicMock(spec=LLMClient)
    mock_skills = MagicMock(spec=SkillRegistry)

    mock_planner = MagicMock(spec=Planner)

    # Needs to return something on the first check of while loop, then []
    mock_planner.get_current_plan.side_effect = [
        [{"skill": "test_mcp.run", "skill_kwargs": {"data": "123"}, "description": "test_step"}],
        [{"skill": "test_mcp.run", "skill_kwargs": {"data": "123"}, "description": "test_step"}],
        []
    ]

    mock_planner.get_completed_steps.return_value = []
    def mock_mark_completed(idx, result_str, user_id=None):
        mock_planner.get_completed_steps.return_value.append({"skill": "test_mcp.run", "result": result_str, "description": "test_step"})

    mock_planner.mark_step_completed.side_effect = mock_mark_completed

    mcp_client = MCPClient()
    mcp_client.register_remote_tool("test_mcp.run", {"url": "mock"})

    agent = GeneratorAgent(
        llm=mock_llm,
        skills=mock_skills,
        planner=mock_planner,
    )
    agent.mcp_client = mcp_client

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"jsonrpc": "2.0", "result": "Executed remote MCP tool test_mcp.run", "id": "123"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = await agent.execute_plan("test input")
        assert "Executed remote MCP tool test_mcp.run" in result

@pytest.mark.asyncio
async def test_generator_agent_mcp_timeout():
    mock_llm = MagicMock(spec=LLMClient)
    mock_skills = MagicMock(spec=SkillRegistry)
    mock_planner = MagicMock(spec=Planner)

    mock_planner.get_current_plan.side_effect = [
        [{"skill": "slow_mcp.tool", "skill_kwargs": {}, "description": "slow_step"}],
        [{"skill": "slow_mcp.tool", "skill_kwargs": {}, "description": "slow_step"}],
        []
    ]

    mock_planner.get_completed_steps.return_value = []
    def mock_mark_completed(idx, result_str, user_id=None):
        mock_planner.get_completed_steps.return_value.append({"skill": "slow_mcp.tool", "result": result_str, "description": "slow_step"})

    mock_planner.mark_step_completed.side_effect = mock_mark_completed

    mcp_client = MCPClient()
    mcp_client.register_remote_tool("slow_mcp.tool", {})

    async def slow_execute(*args, **kwargs):
        await asyncio.sleep(2.0)
        return "too late"

    mcp_client.execute_tool = AsyncMock(side_effect=slow_execute)

    agent = GeneratorAgent(
        llm=mock_llm,
        skills=mock_skills,
        planner=mock_planner,
    )
    agent.mcp_client = mcp_client

    with patch("magda_agent.agents.generator_agent.asyncio.wait_for") as mock_wait:
        mock_wait.side_effect = asyncio.TimeoutError()
        result = await agent.execute_plan("test input")

    assert "timed out" in result
