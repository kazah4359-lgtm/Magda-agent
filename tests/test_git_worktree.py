import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from magda_agent.isolation.git_worktree import GitWorktreeManager
from magda_agent.agents.subagent import SubAgent
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_create_worktree_async() -> None:
    """Test git worktree creation command parsing and isolated directory logic."""
    manager = GitWorktreeManager(base_dir="/tmp/test_worktrees")

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        worktree_path = await manager.create_worktree_async(branch_name="test-branch")

        assert "test_worktrees/worktree_" in worktree_path
        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "git" in args
        assert "worktree" in args
        assert "add" in args

@pytest.mark.asyncio
async def test_remove_worktree_async() -> None:
    """Test git worktree removal fallback and command sequence."""
    manager = GitWorktreeManager(base_dir="/tmp/test_worktrees")
    worktree_path = "/tmp/test_worktrees/worktree_123"

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        with patch("os.path.exists", return_value=False):
            await manager.remove_worktree_async(worktree_path)

            mock_exec.assert_called_once()
            args = mock_exec.call_args[0]
            assert "remove" in args
            assert worktree_path in args

@pytest.mark.asyncio
async def test_subagent_isolation_lifecycle() -> None:
    """Test SubAgent invokes LLM safely while wrapping execution within the isolation boundaries."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_completion = AsyncMock(return_value="Isolated task complete.")

    subagent = SubAgent(llm=mock_llm, use_isolation=True)

    mock_worktree_path = "/tmp/fake_worktree"

    with patch.object(subagent.worktree_manager, 'create_worktree_async', return_value=mock_worktree_path) as mock_create:
        with patch.object(subagent.worktree_manager, 'remove_worktree_async', new_callable=AsyncMock) as mock_remove:
            result = await subagent.execute(task="Do something", context="Some context")

            assert result == "Isolated task complete."
            mock_create.assert_awaited_once()
            mock_remove.assert_awaited_once_with(mock_worktree_path)
            mock_llm.chat_completion.assert_awaited_once()

@pytest.mark.asyncio
async def test_git_worktree_context_manager() -> None:
    """Test GitWorktreeManager's isolated_environment async context manager properly handles setup and teardown."""
    manager = GitWorktreeManager(base_dir="/tmp/test_worktrees")

    mock_worktree_path = "/tmp/test_worktrees/worktree_ctx"

    with patch.object(manager, 'create_worktree_async', return_value=mock_worktree_path) as mock_create:
        with patch.object(manager, 'remove_worktree_async', new_callable=AsyncMock) as mock_remove:
            async with manager.isolated_environment() as path:
                assert path == mock_worktree_path
                mock_create.assert_awaited_once()
                mock_remove.assert_not_called()

            mock_remove.assert_awaited_once_with(mock_worktree_path)
