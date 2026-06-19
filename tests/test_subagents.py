import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from magda_agent.agents.subagents import SubagentsManager
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_subagents_manager_spawn_and_execute_parallel():
    """
    Tests that SubagentsManager correctly spawns sub-agents and executes them in parallel.
    """
    llm_mock = MagicMock(spec=LLMClient)
    manager = SubagentsManager(llm=llm_mock)

    tasks = [
        {"description": "Task 1: Search logs", "system_prompt": "You are a log searcher."},
        {"description": "Task 2: Analyze performance"},
        {"description": "Task 3: Write report"}
    ]

    with patch('magda_agent.agents.subagents.SubAgent') as MockSubAgent:
        # We need to mock the execute method of the instantiated SubAgent
        mock_sub_agent_instance = MagicMock()
        MockSubAgent.return_value = mock_sub_agent_instance

        async def mock_execute(task, context):
            await asyncio.sleep(0.01) # Sleep to simulate work and parallel execution
            return f"Result for {task}"

        mock_sub_agent_instance.execute = AsyncMock(side_effect=mock_execute)

        results = await manager.spawn_and_execute_parallel(tasks=tasks, context="System Context")

        assert len(results) == 3
        assert results == [
            "Result for Task 1: Search logs",
            "Result for Task 2: Analyze performance",
            "Result for Task 3: Write report"
        ]

        assert MockSubAgent.call_count == 3

        # Verify that SubAgent was instantiated with use_isolation=True
        # Task 1
        MockSubAgent.assert_any_call(llm=llm_mock, system_prompt="You are a log searcher.", use_isolation=True)
        # Task 2
        MockSubAgent.assert_any_call(llm=llm_mock, system_prompt=None, use_isolation=True)

        assert mock_sub_agent_instance.execute.call_count == 3
        mock_sub_agent_instance.execute.assert_any_call(task="Task 1: Search logs", context="System Context")
        mock_sub_agent_instance.execute.assert_any_call(task="Task 2: Analyze performance", context="System Context")
        mock_sub_agent_instance.execute.assert_any_call(task="Task 3: Write report", context="System Context")

@pytest.mark.asyncio
async def test_subagents_manager_empty_tasks():
    """
    Tests SubagentsManager with an empty task list.
    """
    llm_mock = MagicMock(spec=LLMClient)
    manager = SubagentsManager(llm=llm_mock)

    results = await manager.spawn_and_execute_parallel(tasks=[], context="System Context")
    assert results == []
