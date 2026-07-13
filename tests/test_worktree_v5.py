import pytest
import os
import shutil
from unittest.mock import AsyncMock, patch
from magda_agent.multi_agent.worktree_v5 import WorktreeManagerV5

@pytest.mark.asyncio
async def test_worktree_v5_create() -> None:
    """Test WorktreeManagerV5 creation command sequence."""
    manager = WorktreeManagerV5(base_dir="/tmp/test_worktree_v5")

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        worktree_path = await manager.create_worktree(agent_id="test-agent", branch_name="test-branch")

        assert "/tmp/test_worktree_v5/worktree_test-agent_" in worktree_path
        assert worktree_path in manager.active_worktrees

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "git" == args[0]
        assert "worktree" == args[1]
        assert "add" == args[2]
        assert "-b" == args[3]
        assert "test-branch" == args[4]
        assert worktree_path == args[5]

@pytest.mark.asyncio
async def test_worktree_v5_remove() -> None:
    """Test WorktreeManagerV5 removal and directory cleanup."""
    manager = WorktreeManagerV5(base_dir="/tmp/test_worktree_v5")
    worktree_path = "/tmp/test_worktree_v5/worktree_test-agent_123"
    manager.active_worktrees[worktree_path] = "test-agent"

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        with patch("os.path.exists", return_value=True):
            with patch("shutil.rmtree") as mock_rmtree:
                await manager.remove_worktree(worktree_path)

                mock_exec.assert_called_once()
                args = mock_exec.call_args[0]
                assert "remove" in args
                assert "--force" in args
                assert worktree_path in args

                mock_rmtree.assert_called_once_with(worktree_path)
                assert worktree_path not in manager.active_worktrees

@pytest.mark.asyncio
async def test_worktree_v5_context_manager() -> None:
    """Test the isolated_environment async context manager."""
    manager = WorktreeManagerV5(base_dir="/tmp/test_worktree_v5")
    worktree_path = "/tmp/test_worktree_v5/worktree_ctx_123"

    with patch.object(manager, 'create_worktree', return_value=worktree_path) as mock_create:
        with patch.object(manager, 'remove_worktree', new_callable=AsyncMock) as mock_remove:
            async with manager.isolated_environment(agent_id="ctx-agent") as path:
                assert path == worktree_path
                mock_create.assert_awaited_once_with(agent_id="ctx-agent", branch_name=None)
                mock_remove.assert_not_called()

            mock_remove.assert_awaited_once_with(worktree_path)

@pytest.mark.asyncio
async def test_worktree_v5_cleanup_all() -> None:
    """Test cleanup_all removes multiple worktrees."""
    manager = WorktreeManagerV5(base_dir="/tmp/test_worktree_v5")
    path1 = "/tmp/test_worktree_v5/w1"
    path2 = "/tmp/test_worktree_v5/w2"
    manager.active_worktrees[path1] = "a1"
    manager.active_worktrees[path2] = "a2"

    with patch.object(manager, 'remove_worktree', new_callable=AsyncMock) as mock_remove:
        await manager.cleanup_all()
        assert mock_remove.call_count == 2
        mock_remove.assert_any_call(path1)
        mock_remove.assert_any_call(path2)
