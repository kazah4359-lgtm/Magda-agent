import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from magda_agent.agents.parallel_subagents_v5 import ParallelSubagentSpawnerV5

@pytest_asyncio.fixture
def mock_llm_client():
    client = AsyncMock()
    # Need to mock the chat_completion inside SubAgent.execute which uses llm.chat_completion
    # Since Subagent calls llm.chat_completion
    client.chat_completion = AsyncMock(return_value="Mocked response")
    return client

@pytest_asyncio.fixture
def mock_isolation_manager():
    manager = AsyncMock()
    manager.create_worktree = AsyncMock(return_value="/tmp/mock_worktree")
    manager.remove_worktree = AsyncMock()
    return manager

@pytest.mark.asyncio
async def test_parallel_spawning_success(mock_llm_client, mock_isolation_manager):
    spawner = ParallelSubagentSpawnerV5(llm=mock_llm_client, isolation_manager=mock_isolation_manager)
    tasks = [
        {"description": "Task 1", "system_prompt": "Sys 1"},
        {"description": "Task 2"}
    ]

    with patch("magda_agent.agents.sub_agent.SubAgent.execute", new_callable=AsyncMock) as mock_execute:
        mock_execute.side_effect = ["Result 1", "Result 2"]
        results = await spawner.run_parallel_tasks(tasks=tasks, base_context="Base context", use_isolation=False)

        assert results == ["Result 1", "Result 2"]
        assert mock_execute.call_count == 2

        # Verify isolation was NOT called
        mock_isolation_manager.create_worktree.assert_not_called()
        mock_isolation_manager.remove_worktree.assert_not_called()

@pytest.mark.asyncio
async def test_worktree_isolation_mocked(mock_llm_client, mock_isolation_manager):
    spawner = ParallelSubagentSpawnerV5(llm=mock_llm_client, isolation_manager=mock_isolation_manager)
    tasks = [{"description": "Isolated Task"}]

    with patch("magda_agent.agents.sub_agent.SubAgent.execute", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "Isolated Result"
        results = await spawner.run_parallel_tasks(tasks=tasks, base_context="Base", use_isolation=True)

        assert results == ["Isolated Result"]
        assert mock_isolation_manager.create_worktree.call_count == 1
        assert mock_isolation_manager.remove_worktree.call_count == 1

        # Verify context included worktree path
        call_args = mock_execute.call_args[1]
        assert "Base\n\nIsolated Git Worktree Path: /tmp/mock_worktree" in call_args['context']
