import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.agents.teams import TeamManager
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_team_manager_spawn_and_execute_parallel():
    """
    Tests that TeamManager spawns sub-agents and executes them in parallel.
    """
    llm_mock = MagicMock(spec=LLMClient)

    manager = TeamManager(llm=llm_mock)

    tasks = [
        {"description": "Task 1: Search logs"},
        {"description": "Task 2: Analyze performance"},
        {"description": "Task 3: Write report"}
    ]

    with patch('magda_agent.agents.teams.SubAgent.execute', new_callable=AsyncMock) as mock_execute:
        # Mock responses based on the task description
        async def mock_execute_side_effect(task, context):
            # Sleep to simulate some work and allow testing of parallel execution
            await asyncio.sleep(0.01)
            return f"Result for {task}"

        mock_execute.side_effect = mock_execute_side_effect

        results = await manager.spawn_and_execute(tasks=tasks, context="System Context")

        assert len(results) == 3
        assert results == [
            "Result for Task 1: Search logs",
            "Result for Task 2: Analyze performance",
            "Result for Task 3: Write report"
        ]

        assert mock_execute.call_count == 3
        # Verify the context is passed to execute
        mock_execute.assert_any_call(task="Task 1: Search logs", context="System Context")
        mock_execute.assert_any_call(task="Task 2: Analyze performance", context="System Context")
        mock_execute.assert_any_call(task="Task 3: Write report", context="System Context")

@pytest.mark.asyncio
async def test_team_manager_empty_tasks():
    """
    Tests TeamManager with an empty task list.
    """
    llm_mock = MagicMock(spec=LLMClient)
    manager = TeamManager(llm=llm_mock)

    results = await manager.spawn_and_execute(tasks=[], context="System Context")
    assert results == []
