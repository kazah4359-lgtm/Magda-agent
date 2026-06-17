
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from magda_agent.agents.team_orchestrator import TeamOrchestrator
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_parallel_execute_success():
    """Tests that TeamOrchestrator spawns parallel sub-agents correctly using SubagentSpawner."""
    llm_mock = MagicMock(spec=LLMClient)

    with patch('magda_agent.agents.team_orchestrator.SubagentSpawner') as MockSpawner:
        mock_spawner_instance = MockSpawner.return_value
        mock_spawner_instance.spawn_and_execute = AsyncMock(return_value=["Result 1", "Result 2"])

        orchestrator = TeamOrchestrator(llm=llm_mock)

        tasks = [
            {"description": "Task 1"},
            {"description": "Task 2"}
        ]

        results = await orchestrator.parallel_execute(tasks, base_context="Context")

        assert len(results) == 2
        assert results[0] == "Result 1"
        assert results[1] == "Result 2"
        mock_spawner_instance.spawn_and_execute.assert_called_once_with(tasks, "Context", use_isolation=True)
