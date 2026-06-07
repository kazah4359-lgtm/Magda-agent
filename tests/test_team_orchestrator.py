import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.agents.team_orchestrator import TeamOrchestrator
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_parallel_execute_success():
    """Tests that TeamOrchestrator spawns parallel sub-agents correctly."""
    llm_mock = MagicMock(spec=LLMClient)
    orchestrator = TeamOrchestrator(llm=llm_mock)

    tasks = [
        {"description": "Task 1"},
        {"description": "Task 2"}
    ]

    with patch('magda_agent.agents.team_orchestrator.SubAgent') as MockSubAgent:
        mock_instance_1 = MagicMock()
        mock_instance_1.execute = AsyncMock(return_value="Result 1")

        mock_instance_2 = MagicMock()
        mock_instance_2.execute = AsyncMock(return_value="Result 2")

        # Return mock_instance_1 first, then mock_instance_2
        MockSubAgent.side_effect = [mock_instance_1, mock_instance_2]

        results = await orchestrator.parallel_execute(tasks, base_context="Context")

        assert len(results) == 2
        assert "Result 1" in results
        assert "Result 2" in results

        assert MockSubAgent.call_count == 2
        MockSubAgent.assert_called_with(llm=llm_mock, use_isolation=True)

        mock_instance_1.execute.assert_awaited_once_with(task="Task 1", context="Context")
        mock_instance_2.execute.assert_awaited_once_with(task="Task 2", context="Context")
