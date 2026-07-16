import pytest
import os
import shutil
import asyncio
from unittest.mock import AsyncMock, patch
from magda_agent.agents.worktree_manager_v10 import WorktreeManagerV10

@pytest.mark.asyncio
async def test_create_worktree_detached() -> None:
    """Test WorktreeManagerV10 creates a detached worktree correctly when no branch is provided."""
    manager = WorktreeManagerV10(base_dir="/tmp/test_worktrees_v10")

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        worktree_path = await manager.create_worktree(agent_id="agent-1")

        assert "/tmp/test_worktrees_v10/worktree_agent-1_" in worktree_path
        assert worktree_path in manager.active_worktrees
        assert manager.active_worktrees[worktree_path] == "agent-1"

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "git" == args[0]
        assert "worktree" == args[1]
        assert "add" == args[2]
        assert "-d" == args[3]
        assert worktree_path == args[4]
        assert "HEAD" == args[5]

@pytest.mark.asyncio
async def test_create_worktree_branch() -> None:
    """Test WorktreeManagerV10 creates a branch-based worktree correctly when branch name is provided."""
    manager = WorktreeManagerV10(base_dir="/tmp/test_worktrees_v10")

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        worktree_path = await manager.create_worktree(agent_id="agent-2", branch_name="feature-v10")

        assert "/tmp/test_worktrees_v10/worktree_agent-2_" in worktree_path
        assert worktree_path in manager.active_worktrees
        assert manager.active_worktrees[worktree_path] == "agent-2"

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "git" == args[0]
        assert "worktree" == args[1]
        assert "add" == args[2]
        assert "-b" == args[3]
        assert "feature-v10" == args[4]
        assert worktree_path == args[5]
        assert "HEAD" == args[6]

@pytest.mark.asyncio
async def test_create_worktree_failure() -> None:
    """Test WorktreeManagerV10 raises a RuntimeError if the git command fails."""
    manager = WorktreeManagerV10(base_dir="/tmp/test_worktrees_v10")

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"fatal: error")
    mock_process.returncode = 128

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="Git worktree creation failed"):
            await manager.create_worktree()

@pytest.mark.asyncio
async def test_remove_worktree_success() -> None:
    """Test WorktreeManagerV10 successfully removes a worktree and tracks state cleanly."""
    manager = WorktreeManagerV10(base_dir="/tmp/test_worktrees_v10")
    worktree_path = "/tmp/test_worktrees_v10/worktree_agent_123"
    manager.active_worktrees[worktree_path] = "agent_123"

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        with patch("os.path.exists", return_value=True) as mock_exists:
            with patch("shutil.rmtree") as mock_rmtree:
                await manager.remove_worktree(worktree_path)

                mock_exec.assert_called_once()
                args = mock_exec.call_args[0]
                assert "remove" in args
                assert "--force" in args
                assert worktree_path in args

                mock_exists.assert_called_once_with(worktree_path)
                mock_rmtree.assert_called_once_with(worktree_path)
                assert worktree_path not in manager.active_worktrees

@pytest.mark.asyncio
async def test_remove_worktree_fallback() -> None:
    """Test WorktreeManagerV10 fallback to manual rmtree cleanup when git command fails."""
    manager = WorktreeManagerV10(base_dir="/tmp/test_worktrees_v10")
    worktree_path = "/tmp/test_worktrees_v10/worktree_agent_456"
    manager.active_worktrees[worktree_path] = "agent_456"

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"Error removal")
    mock_process.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with patch("os.path.exists", return_value=True):
            with patch("shutil.rmtree") as mock_rmtree:
                await manager.remove_worktree(worktree_path)

                mock_rmtree.assert_called_once_with(worktree_path)
                assert worktree_path not in manager.active_worktrees

@pytest.mark.asyncio
async def test_isolated_environment() -> None:
    """Test isolated_environment async context manager sets up and removes worktree path correctly."""
    manager = WorktreeManagerV10(base_dir="/tmp/test_worktrees_v10")
    mock_worktree_path = "/tmp/test_worktrees_v10/worktree_ctx"

    with patch.object(manager, 'create_worktree', return_value=mock_worktree_path) as mock_create:
        with patch.object(manager, 'remove_worktree', new_callable=AsyncMock) as mock_remove:
            async with manager.isolated_environment(agent_id="ctx-agent") as path:
                assert path == mock_worktree_path
                mock_create.assert_awaited_once_with(agent_id="ctx-agent", branch_name=None)
                mock_remove.assert_not_called()

            mock_remove.assert_awaited_once_with(mock_worktree_path)

@pytest.mark.asyncio
async def test_execute_in_isolation_success() -> None:
    """Test execute_in_isolation runs task successfully and does clean up."""
    manager = WorktreeManagerV10(base_dir="/tmp/test_worktrees_v10")
    mock_worktree_path = "/tmp/test_worktrees_v10/worktree_exec"

    async def mock_task(path: str) -> str:
        return f"Completed in {path}"

    with patch.object(manager, 'create_worktree', return_value=mock_worktree_path) as mock_create:
        with patch.object(manager, 'remove_worktree', new_callable=AsyncMock) as mock_remove:
            result = await manager.execute_in_isolation(mock_task, agent_id="exec-agent")

            assert result == f"Completed in {mock_worktree_path}"
            mock_create.assert_awaited_once_with(agent_id="exec-agent", branch_name=None)
            mock_remove.assert_awaited_once_with(mock_worktree_path)

@pytest.mark.asyncio
async def test_execute_in_isolation_timeout() -> None:
    """Test execute_in_isolation throws a TimeoutError and cleans up when timeout is reached."""
    manager = WorktreeManagerV10(base_dir="/tmp/test_worktrees_v10")
    mock_worktree_path = "/tmp/test_worktrees_v10/worktree_timeout"

    async def mock_slow_task(path: str) -> str:
        await asyncio.sleep(2)
        return "Not completed"

    with patch.object(manager, 'create_worktree', return_value=mock_worktree_path) as mock_create:
        with patch.object(manager, 'remove_worktree', new_callable=AsyncMock) as mock_remove:
            with pytest.raises(asyncio.TimeoutError):
                await manager.execute_in_isolation(mock_slow_task, timeout=0.01)

            mock_create.assert_awaited_once()
            mock_remove.assert_awaited_once_with(mock_worktree_path)

def test_get_active_worktrees() -> None:
    """Test get_active_worktrees returns a copy of tracked worktrees dict."""
    manager = WorktreeManagerV10(base_dir="/tmp/test_worktrees_v10")
    manager.active_worktrees["/path1"] = "agent1"
    manager.active_worktrees["/path2"] = "agent2"

    active = manager.get_active_worktrees()
    assert active == {"/path1": "agent1", "/path2": "agent2"}
    # Assert mutation of copy does not modify internal state
    active["/path3"] = "agent3"
    assert "/path3" not in manager.active_worktrees

def test_verify_sandbox() -> None:
    """Test verify_sandbox correctly evaluates sandbox safety constraints."""
    manager = WorktreeManagerV10(base_dir="/tmp/test_worktrees_v10")
    valid_path = "/tmp/test_worktrees_v10/worktree_agent1_abc"
    invalid_path = "/etc/passwd"
    nonexistent_path = "/tmp/test_worktrees_v10/does_not_exist"

    with patch("os.path.exists") as mock_exists:
        # 1. Non-existent path
        mock_exists.return_value = False
        assert not manager.verify_sandbox(nonexistent_path)

        # 2. Existing path under base_dir
        mock_exists.return_value = True
        assert manager.verify_sandbox(valid_path)

        # 3. Existing path NOT under base_dir
        assert not manager.verify_sandbox(invalid_path)

@pytest.mark.asyncio
async def test_cleanup_all() -> None:
    """Test cleanup_all removes all active worktrees."""
    manager = WorktreeManagerV10(base_dir="/tmp/test_worktrees_v10")
    manager.active_worktrees = {
        "/path1": "a1",
        "/path2": "a2"
    }

    with patch.object(manager, 'remove_worktree', new_callable=AsyncMock) as mock_remove:
        await manager.cleanup_all()
        assert mock_remove.call_count == 2
        mock_remove.assert_any_call("/path1")
        mock_remove.assert_any_call("/path2")
