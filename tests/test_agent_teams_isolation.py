import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.agents.sub_agent import SubAgent
from magda_agent.llm_client import LLMClient
from magda_agent.isolation.git_worktree import GitWorktreeManager

@pytest.mark.asyncio
async def test_sub_agent_with_isolation_success():
    """Tests SubAgent successfully uses and cleans up an isolated git worktree."""
    llm_mock = MagicMock(spec=LLMClient)
    llm_mock.chat_completion = AsyncMock(return_value="Task completed in isolation.")

    with patch('magda_agent.isolation.git_worktree.GitWorktreeManager.create_worktree_async', new_callable=AsyncMock) as mock_create, \
         patch('magda_agent.isolation.git_worktree.GitWorktreeManager.remove_worktree_async', new_callable=AsyncMock) as mock_remove:

        mock_create.return_value = "/tmp/magda_worktrees/worktree_mock123"

        sub_agent = SubAgent(llm=llm_mock, use_isolation=True)
        result = await sub_agent.execute(task="Test task", context="Base context")

        assert result == "Task completed in isolation."

        # Verify worktree creation
        mock_create.assert_awaited_once()

        # Verify context injection
        llm_mock.chat_completion.assert_awaited_once()
        call_args = llm_mock.chat_completion.call_args[0][0]
        user_message = call_args[1]["content"]
        assert "Base context" in user_message
        assert "/tmp/magda_worktrees/worktree_mock123" in user_message
        assert "Test task" in user_message

        # Verify cleanup
        mock_remove.assert_awaited_once_with("/tmp/magda_worktrees/worktree_mock123")

@pytest.mark.asyncio
async def test_sub_agent_with_isolation_llm_failure():
    """Tests SubAgent cleans up worktree even if LLM execution fails."""
    llm_mock = MagicMock(spec=LLMClient)
    llm_mock.chat_completion = AsyncMock(side_effect=Exception("LLM API Error"))

    with patch('magda_agent.isolation.git_worktree.GitWorktreeManager.create_worktree_async', new_callable=AsyncMock) as mock_create, \
         patch('magda_agent.isolation.git_worktree.GitWorktreeManager.remove_worktree_async', new_callable=AsyncMock) as mock_remove:

        mock_create.return_value = "/tmp/magda_worktrees/worktree_mock456"

        sub_agent = SubAgent(llm=llm_mock, use_isolation=True)
        result = await sub_agent.execute(task="Test task", context="Base context")

        assert "Error executing SubAgent task: LLM API Error" in result

        mock_create.assert_awaited_once()
        llm_mock.chat_completion.assert_awaited_once()

        # Verify cleanup happens despite error
        mock_remove.assert_awaited_once_with("/tmp/magda_worktrees/worktree_mock456")

@pytest.mark.asyncio
async def test_sub_agent_with_isolation_creation_failure():
    """Tests SubAgent behavior when worktree creation fails."""
    llm_mock = MagicMock(spec=LLMClient)

    with patch('magda_agent.isolation.git_worktree.GitWorktreeManager.create_worktree_async', new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = RuntimeError("Git worktree creation failed")

        sub_agent = SubAgent(llm=llm_mock, use_isolation=True)
        result = await sub_agent.execute(task="Test task", context="Base context")

        assert "Error: Failed to create isolated worktree" in result
        mock_create.assert_awaited_once()
        llm_mock.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_git_worktree_manager_create_success():
    """Tests the GitWorktreeManager creates a worktree."""
    manager = GitWorktreeManager(base_dir="/tmp/test_worktrees")

    process_mock = MagicMock()
    process_mock.communicate = AsyncMock(return_value=(b"Success", b""))
    process_mock.returncode = 0

    with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = process_mock

        path = await manager.create_worktree_async()

        assert path.startswith("/tmp/test_worktrees/worktree_")
        mock_exec.assert_called_once()
        assert "git" in mock_exec.call_args[0]
        assert "worktree" in mock_exec.call_args[0]

@pytest.mark.asyncio
async def test_git_worktree_manager_remove_success():
    """Tests the GitWorktreeManager removes a worktree."""
    manager = GitWorktreeManager(base_dir="/tmp/test_worktrees")

    process_mock = MagicMock()
    process_mock.communicate = AsyncMock(return_value=(b"Success", b""))
    process_mock.returncode = 0

    with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = process_mock

        await manager.remove_worktree_async("/tmp/test_worktrees/worktree_abc123")

        mock_exec.assert_called_once()
        assert "remove" in mock_exec.call_args[0]
        assert "/tmp/test_worktrees/worktree_abc123" in mock_exec.call_args[0]
