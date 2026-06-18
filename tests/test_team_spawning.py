import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.agents.team_spawning import ContextAwareTeamSpawner
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_context_aware_team_spawner():
    mock_llm = MagicMock(spec=LLMClient)
    mock_context_engine = MagicMock(spec=ContextEngine)

    mock_context_engine.retrieve_context.return_value = ["base_context_item"]
    mock_context_engine.assemble = AsyncMock(return_value="assembled_context")

    spawner = ContextAwareTeamSpawner(llm=mock_llm, context_engine=mock_context_engine)

    # Mocking the internal SubagentSpawner logic
    spawner.spawner = MagicMock()
    spawner.spawner.spawn_and_execute = AsyncMock(return_value=["mocked_result"])

    tasks = [{"description": "test task"}]
    results = await spawner.spawn_and_execute(tasks=tasks, base_context="base", user_id=1)

    assert results == ["mocked_result"]
    mock_context_engine.retrieve_context.assert_called_once()
    mock_context_engine.assemble.assert_called_once()
    spawner.spawner.spawn_and_execute.assert_called_once_with(
        tasks=[{"description": "test task"}],
        base_context="assembled_context",
        use_isolation=True
    )
