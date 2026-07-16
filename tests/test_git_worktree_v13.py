import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from magda_agent.isolation.git_worktree_v13 import GitWorktreeManagerV13

@pytest.mark.asyncio
async def test_create_worktree_async_detached() -> None:
    """Test GitWorktreeManagerV13 creates a detached worktree correctly when no branch is provided."""
    manager = GitWorktreeManagerV13(base_dir="/tmp/test_worktrees_v13")

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        worktree_path = await manager.create_worktree_async()

        assert "test_worktrees_v13/worktree_" in worktree_path
        assert worktree_path in manager.active_worktrees
        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "git" in args
        assert "worktree" in args
        assert "add" in args
        assert "-d" in args
        assert "HEAD" in args

@pytest.mark.asyncio
async def test_create_worktree_async_branch() -> None:
    """Test GitWorktreeManagerV13 creates a branch-based worktree correctly when branch name is provided."""
    manager = GitWorktreeManagerV13(base_dir="/tmp/test_worktrees_v13")

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        worktree_path = await manager.create_worktree_async(branch_name="feature-v13")

        assert "test_worktrees_v13/worktree_" in worktree_path
        assert worktree_path in manager.active_worktrees
        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "git" in args
        assert "worktree" in args
        assert "add" in args
        assert "-b" in args
        assert "feature-v13" in args
        assert "HEAD" in args

@pytest.mark.asyncio
async def test_remove_worktree_async_success() -> None:
    """Test GitWorktreeManagerV13 successfully removes worktree path via git and cleans up tracking state."""
    manager = GitWorktreeManagerV13(base_dir="/tmp/test_worktrees_v13")
    worktree_path = "/tmp/test_worktrees_v13/worktree_123"
    manager.active_worktrees.add(worktree_path)

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        await manager.remove_worktree_async(worktree_path)

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "remove" in args
        assert "--force" in args
        assert worktree_path in args
        assert worktree_path not in manager.active_worktrees

@pytest.mark.asyncio
async def test_remove_worktree_async_failure_fallback() -> None:
    """Test GitWorktreeManagerV13 falls back to manual folder deletion when git worktree remove command fails."""
    manager = GitWorktreeManagerV13(base_dir="/tmp/test_worktrees_v13")
    worktree_path = "/tmp/test_worktrees_v13/worktree_123"
    manager.active_worktrees.add(worktree_path)

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"Failed to delete")
    mock_process.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        with patch("os.path.exists", return_value=True) as mock_exists:
            with patch("shutil.rmtree") as mock_rmtree:
                await manager.remove_worktree_async(worktree_path)

                mock_exec.assert_called_once()
                mock_exists.assert_called_with(worktree_path)
                mock_rmtree.assert_called_with(worktree_path)
                assert worktree_path not in manager.active_worktrees

@pytest.mark.asyncio
async def test_isolated_environment_lifecycle() -> None:
    """Test isolated_environment context manager properly runs tasks and cleans up worktree path."""
    manager = GitWorktreeManagerV13(base_dir="/tmp/test_worktrees_v13")
    mock_worktree_path = "/tmp/test_worktrees_v13/worktree_abc"

    with patch.object(manager, 'create_worktree_async', return_value=mock_worktree_path) as mock_create:
        with patch.object(manager, 'remove_worktree_async', new_callable=AsyncMock) as mock_remove:
            async with manager.isolated_environment() as path:
                assert path == mock_worktree_path
                mock_create.assert_awaited_once()
                mock_remove.assert_not_called()

            mock_remove.assert_awaited_once_with(mock_worktree_path)

@pytest.mark.asyncio
async def test_execute_in_isolation_success() -> None:
    """Test execute_in_isolation executes task and completes cleanly."""
    manager = GitWorktreeManagerV13(base_dir="/tmp/test_worktrees_v13")
    mock_worktree_path = "/tmp/test_worktrees_v13/worktree_xyz"

    async def mock_task(path: str) -> str:
        return f"Completed in {path}"

    with patch.object(manager, 'create_worktree_async', return_value=mock_worktree_path) as mock_create:
        with patch.object(manager, 'remove_worktree_async', new_callable=AsyncMock) as mock_remove:
            result = await manager.execute_in_isolation(mock_task)

            assert result == f"Completed in {mock_worktree_path}"
            mock_create.assert_awaited_once()
            mock_remove.assert_awaited_once_with(mock_worktree_path)

@pytest.mark.asyncio
async def test_execute_in_isolation_timeout() -> None:
    """Test execute_in_isolation raises TimeoutError and cleans up worktree path when timeout limits are breached."""
    manager = GitWorktreeManagerV13(base_dir="/tmp/test_worktrees_v13")
    mock_worktree_path = "/tmp/test_worktrees_v13/worktree_timeout"

    async def mock_task(path: str) -> str:
        await asyncio.sleep(2)
        return f"Completed in {path}"

    with patch.object(manager, 'create_worktree_async', return_value=mock_worktree_path) as mock_create:
        with patch.object(manager, 'remove_worktree_async', new_callable=AsyncMock) as mock_remove:
            with pytest.raises(asyncio.TimeoutError):
                await manager.execute_in_isolation(mock_task, timeout=0.01)

            mock_create.assert_awaited_once()
            mock_remove.assert_awaited_once_with(mock_worktree_path)
