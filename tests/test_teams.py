import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.agents.teams import TeamManager
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_team_manager_spawn_and_execute() -> None:
    """Tests that TeamManager spawns isolated sub-agents for tasks."""
    llm_mock = MagicMock(spec=LLMClient)
    manager = TeamManager(llm=llm_mock)

    tasks = [{"description": "Task 1"}, {"description": "Task 2"}]

    with patch('magda_agent.agents.teams.SubAgent') as MockSubAgent:
        sub_mock_1 = MagicMock()
        sub_mock_1.execute = AsyncMock(return_value="Result 1")
        sub_mock_2 = MagicMock()
        sub_mock_2.execute = AsyncMock(return_value="Result 2")

        MockSubAgent.side_effect = [sub_mock_1, sub_mock_2]

        results = await manager.spawn_and_execute(tasks, context="Base context")

        assert results == ["Result 1", "Result 2"]
        assert MockSubAgent.call_count == 2
        MockSubAgent.assert_called_with(llm=llm_mock, use_isolation=True)
        sub_mock_1.execute.assert_awaited_once_with(task="Task 1", context="Base context")
        sub_mock_2.execute.assert_awaited_once_with(task="Task 2", context="Base context")
