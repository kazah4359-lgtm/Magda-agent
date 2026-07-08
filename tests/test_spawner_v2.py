import pytest
from unittest.mock import AsyncMock, patch
from magda_agent.agents.spawner_v2 import SubagentSpawnerV2
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_subagent_spawner_v2_success() -> None:
    """Test successful task execution and isolation for multiple tasks."""
    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.chat_completion.side_effect = ["Result 1", "Result 2"]

    with patch("magda_agent.agents.spawner_v2.GitWorktreeIsolationV2") as mock_isolation_class:
        mock_isolation_instance = mock_isolation_class.return_value
        mock_isolation_instance.setup_isolation = AsyncMock(side_effect=["/tmp/wt1", "/tmp/wt2"])
        mock_isolation_instance.teardown_isolation = AsyncMock()

        spawner = SubagentSpawnerV2(llm=mock_llm)
        tasks = [
            {"description": "Task 1", "system_prompt": "Sys 1"},
            {"description": "Task 2", "system_prompt": "Sys 2"}
        ]

        results = await spawner.spawn_and_execute(tasks=tasks, base_context="Shared Context")

        assert len(results) == 2
        assert "Result 1" in results
        assert "Result 2" in results

        assert mock_isolation_instance.setup_isolation.call_count == 2
        assert mock_isolation_instance.teardown_isolation.call_count == 2
        assert mock_llm.chat_completion.call_count == 2

@pytest.mark.asyncio
async def test_subagent_spawner_v2_setup_failure() -> None:
    """Test behavior when isolation setup fails."""
    mock_llm = AsyncMock(spec=LLMClient)

    with patch("magda_agent.agents.spawner_v2.GitWorktreeIsolationV2") as mock_isolation_class:
        mock_isolation_instance = mock_isolation_class.return_value
        mock_isolation_instance.setup_isolation = AsyncMock(side_effect=Exception("Git Error"))
        mock_isolation_instance.teardown_isolation = AsyncMock()

        spawner = SubagentSpawnerV2(llm=mock_llm)
        tasks = [{"description": "Task 1"}]

        results = await spawner.spawn_and_execute(tasks=tasks, base_context="Shared")

        assert len(results) == 1
        assert "Error executing SubAgent task: Git Error" in results[0]

        # Teardown should still be called
        assert mock_isolation_instance.teardown_isolation.call_count == 1
        # Chat completion shouldn't be called
        assert mock_llm.chat_completion.call_count == 0

@pytest.mark.asyncio
async def test_subagent_spawner_v2_execution_failure() -> None:
    """Test behavior when LLM execution fails, ensuring teardown still occurs."""
    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.chat_completion.side_effect = Exception("LLM Error")

    with patch("magda_agent.agents.spawner_v2.GitWorktreeIsolationV2") as mock_isolation_class:
        mock_isolation_instance = mock_isolation_class.return_value
        mock_isolation_instance.setup_isolation = AsyncMock(return_value="/tmp/wt1")
        mock_isolation_instance.teardown_isolation = AsyncMock()

        spawner = SubagentSpawnerV2(llm=mock_llm)
        tasks = [{"description": "Task 1"}]

        results = await spawner.spawn_and_execute(tasks=tasks, base_context="Shared")

        assert len(results) == 1
        assert "Error executing SubAgent task: LLM Error" in results[0]

        assert mock_isolation_instance.setup_isolation.call_count == 1
        assert mock_isolation_instance.teardown_isolation.call_count == 1
