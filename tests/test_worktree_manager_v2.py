import asyncio
import os
import shutil
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from magda_agent.agents.worktree_manager_v2 import WorktreeManagerV2

@pytest.fixture
def base_dir(tmp_path):
    path = str(tmp_path / "test_worktrees")
    os.makedirs(path, exist_ok=True)
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)

@pytest.mark.asyncio
async def test_create_worktree(base_dir):
    """
    Tests that a worktree is created correctly.
    """
    manager = WorktreeManagerV2(base_dir=base_dir)

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        path = await manager.create_worktree("agent123", branch_name="feature-x")

        assert "agent123" in manager.active_worktrees
        assert manager.active_worktrees["agent123"] == path

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "git" in args
        assert "worktree" in args
        assert "add" in args
        assert "-b" in args
        assert "feature-x" in args

@pytest.mark.asyncio
async def test_create_worktree_detached(base_dir):
    """
    Tests that a detached worktree is created correctly.
    """
    manager = WorktreeManagerV2(base_dir=base_dir)

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        path = await manager.create_worktree("agent456")

        assert "agent456" in manager.active_worktrees
        assert manager.active_worktrees["agent456"] == path

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "git" in args
        assert "worktree" in args
        assert "add" in args
        assert "-d" in args

@pytest.mark.asyncio
async def test_create_worktree_failure(base_dir):
    """
    Tests worktree creation failure.
    """
    manager = WorktreeManagerV2(base_dir=base_dir)

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"fatal: error")
    mock_process.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="Git worktree creation failed"):
            await manager.create_worktree("agent123")

@pytest.mark.asyncio
async def test_cleanup_worktree(base_dir):
    """
    Tests that a worktree is removed cleanly.
    """
    manager = WorktreeManagerV2(base_dir=base_dir)

    # Manually setup active worktree state
    test_path = os.path.join(base_dir, "test_agent_path")
    os.makedirs(test_path, exist_ok=True)
    manager.active_worktrees["agent123"] = test_path

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        await manager.cleanup_worktree("agent123")

        assert "agent123" not in manager.active_worktrees
        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "git" in args
        assert "worktree" in args
        assert "remove" in args
        assert "--force" in args
        assert test_path in args

@pytest.mark.asyncio
async def test_cleanup_worktree_fallback(base_dir):
    """
    Tests that a worktree folder is removed even if git command fails.
    """
    manager = WorktreeManagerV2(base_dir=base_dir)

    # Manually setup active worktree state
    test_path = os.path.join(base_dir, "test_agent_path_fallback")
    os.makedirs(test_path, exist_ok=True)
    manager.active_worktrees["agent789"] = test_path

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"fatal: git worktree remove failed")
    mock_process.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec, \
         patch("shutil.rmtree") as mock_rmtree:
        await manager.cleanup_worktree("agent789")

        assert "agent789" not in manager.active_worktrees
        mock_exec.assert_called_once()
        mock_rmtree.assert_called_once_with(test_path)

@pytest.mark.asyncio
async def test_cleanup_all(base_dir):
    """
    Tests that all worktrees are cleaned up.
    """
    manager = WorktreeManagerV2(base_dir=base_dir)
    manager.active_worktrees = {
        "agent1": "/tmp/fake1",
        "agent2": "/tmp/fake2"
    }

    with patch.object(manager, "cleanup_worktree", new_callable=AsyncMock) as mock_cleanup:
        await manager.cleanup_all()

        assert mock_cleanup.call_count == 2
        mock_cleanup.assert_any_call("agent1")
        mock_cleanup.assert_any_call("agent2")