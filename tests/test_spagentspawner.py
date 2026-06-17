import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from magda_agent.agents.spawner import SubagentSpawner
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_subagent_spawner_spawn_and_execute():
    """
    Tests that the SubagentSpawner correctly spawns subagents and executes tasks in parallel.
    """
    mock_llm = MagicMock(spec=LLMClient)
    # The SubAgent internally calls context_compressor which calls LLMClient.chat_completion
    # We mock LLMClient's chat_completion to return a predictable response
    mock_llm.chat_completion = AsyncMock(return_value="Mocked subagent response")

    spawner = SubagentSpawner(llm=mock_llm)

    tasks = [
        {"description": "Task 1", "system_prompt": "Prompt 1"},
        {"description": "Task 2"},
    ]
    base_context = "Base Context"

    # Mock GitWorktreeManager to avoid actual git operations during tests
    with patch("magda_agent.agents.sub_agent.GitWorktreeManager") as MockGitWorktreeManager:
        mock_manager_instance = MockGitWorktreeManager.return_value
        mock_manager_instance.create_worktree_async = AsyncMock(return_value="/tmp/mock_worktree")
        mock_manager_instance.remove_worktree_async = AsyncMock()

        results = await spawner.spawn_and_execute(tasks, base_context=base_context, use_isolation=True)

        assert len(results) == 2
        assert results[0] == "Mocked subagent response"
        assert results[1] == "Mocked subagent response"

        # Check that worktree manager methods were called for both subagents
        assert mock_manager_instance.create_worktree_async.call_count == 2
        assert mock_manager_instance.remove_worktree_async.call_count == 2

@pytest.mark.asyncio
async def test_subagent_spawner_no_isolation():
    """
    Tests that the SubagentSpawner correctly executes tasks when use_isolation is False.
    """
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_completion = AsyncMock(return_value="Mocked response no isolation")

    spawner = SubagentSpawner(llm=mock_llm)

    tasks = [
        {"description": "Task A"},
    ]
    base_context = "Base Context"

    # with use_isolation=False, GitWorktreeManager should not be initialized
    with patch("magda_agent.agents.sub_agent.GitWorktreeManager") as MockGitWorktreeManager:
        results = await spawner.spawn_and_execute(tasks, base_context=base_context, use_isolation=False)

        assert len(results) == 1
        assert results[0] == "Mocked response no isolation"

        # GitWorktreeManager should not be used
        MockGitWorktreeManager.assert_not_called()
