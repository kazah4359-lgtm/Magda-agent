import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.agents.sub_agent import SubAgent
from magda_agent.planning.planner import Planner
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_sub_agent_execute() -> None:
    """Tests that the SubAgent correctly executes a task with the given context."""
    llm_mock = MagicMock(spec=LLMClient)
    llm_mock.chat_completion = AsyncMock(return_value="Sub-agent task completed.")

    sub_agent = SubAgent(llm=llm_mock)
    result = await sub_agent.execute(task="Analyze this code", context="Code is written in Python")

    assert result == "Sub-agent task completed."
    llm_mock.chat_completion.assert_awaited_once()

    call_args = llm_mock.chat_completion.call_args[0][0]
    assert len(call_args) == 2
    assert call_args[0]["role"] == "system"
    assert call_args[1]["role"] == "user"
    assert "Analyze this code" in call_args[1]["content"]
    assert "Code is written in Python" in call_args[1]["content"]

@pytest.mark.asyncio
async def test_planner_spawn_sub_agent() -> None:
    """Tests that the Planner can correctly spawn and execute a sub-agent."""
    llm_mock = MagicMock(spec=LLMClient)
    llm_mock.chat_completion = AsyncMock(return_value="Planner sub-agent completed.")
    skills_mock = MagicMock()

    planner = Planner(llm=llm_mock, skills=skills_mock)
    result = await planner.spawn_sub_agent(task="Write a test", context="Testing framework is pytest")

    assert result == "Planner sub-agent completed."
    llm_mock.chat_completion.assert_awaited_once()
