import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.agents.team_isolation import TeamIsolationOrchestrator
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_execute_isolated_tasks_success():
    """Tests that TeamIsolationOrchestrator spawns parallel sub-agents correctly with isolation."""
    llm_mock = MagicMock(spec=LLMClient)
    orchestrator = TeamIsolationOrchestrator(llm=llm_mock)

    tasks = [
        {"description": "Isolated Task 1"},
        {"description": "Isolated Task 2"}
    ]

    with patch('magda_agent.agents.team_isolation.SubAgent') as MockSubAgent:
        mock_instance_1 = MagicMock()
        mock_instance_1.execute = AsyncMock(return_value="Isolated Result 1")

        mock_instance_2 = MagicMock()
        mock_instance_2.execute = AsyncMock(return_value="Isolated Result 2")

        MockSubAgent.side_effect = [mock_instance_1, mock_instance_2]

        results = await orchestrator.execute_isolated_tasks(tasks, base_context="Isolation Context")

        assert len(results) == 2
        assert "Isolated Result 1" in results
        assert "Isolated Result 2" in results

        assert MockSubAgent.call_count == 2
        MockSubAgent.assert_called_with(llm=llm_mock, use_isolation=True)

        mock_instance_1.execute.assert_awaited_once_with(task="Isolated Task 1", context="Isolation Context")
        mock_instance_2.execute.assert_awaited_once_with(task="Isolated Task 2", context="Isolation Context")
