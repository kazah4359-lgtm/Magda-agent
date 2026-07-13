import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from magda_agent.agents.worktree_dispatcher_v6 import WorktreeDispatcherV6
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_dispatch_tasks_success():
    # Mock LLMClient
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_completion = AsyncMock(return_value="Task completed successfully")

    # Mock GitWorktreeManager to avoid actual git commands
    with patch("magda_agent.agents.worktree_dispatcher_v6.GitWorktreeManager") as MockManager:
        mock_instance = MockManager.return_value
        # isolated_environment is an async context manager
        mock_instance.isolated_environment.return_value.__aenter__ = AsyncMock(return_value="/tmp/mock_worktree")
        mock_instance.isolated_environment.return_value.__aexit__ = AsyncMock(return_value=None)

        dispatcher = WorktreeDispatcherV6(llm=mock_llm)

        tasks = [
            {"description": "Task 1", "system_prompt": "Prompt 1"},
            {"description": "Task 2", "system_prompt": "Prompt 2"}
        ]
        base_context = "Some base context"

        results = await dispatcher.dispatch_tasks(tasks, base_context)

        assert len(results) == 2
        assert all(r == "Task completed successfully" for r in results)
        assert mock_llm.chat_completion.call_count == 2
        assert mock_instance.isolated_environment.call_count == 2

@pytest.mark.asyncio
async def test_dispatch_tasks_isolation_failure():
    # Mock LLMClient
    mock_llm = MagicMock(spec=LLMClient)

    # Mock GitWorktreeManager to raise an exception
    with patch("magda_agent.agents.worktree_dispatcher_v6.GitWorktreeManager") as MockManager:
        mock_instance = MockManager.return_value
        mock_instance.isolated_environment.return_value.__aenter__ = AsyncMock(side_effect=Exception("Git failure"))

        dispatcher = WorktreeDispatcherV6(llm=mock_llm)

        tasks = [{"description": "Task 1"}]
        results = await dispatcher.dispatch_tasks(tasks, "context")

        assert len(results) == 1
        assert "Error: Dispatch failed - Git failure" in results[0]
