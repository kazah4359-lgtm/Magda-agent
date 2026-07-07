import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import os
from magda_agent.agents.isolation_v2 import GitWorktreeIsolationV2

@pytest.fixture
def isolation_manager():
    return GitWorktreeIsolationV2(base_dir="/tmp/test_magda_worktrees_v2")

@pytest.mark.asyncio
async def test_setup_isolation_success(isolation_manager):
    agent_id = "test_agent_123"

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec, \
         patch("uuid.uuid4", return_value="12345678-abcd"):

        env_path = await isolation_manager.setup_isolation(agent_id)

        assert "subagent_test_agent_123_12345678" in env_path
        assert agent_id in isolation_manager.active_contexts

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert args[0] == "git"
        assert args[1] == "worktree"
        assert args[2] == "add"
        assert args[3] == "-d"
        assert args[4] == env_path
        assert args[5] == "HEAD"

@pytest.mark.asyncio
async def test_setup_isolation_with_branch(isolation_manager):
    agent_id = "test_agent_branch"
    branch_name = "feature-branch"

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec, \
         patch("uuid.uuid4", return_value="12345678-abcd"):

        env_path = await isolation_manager.setup_isolation(agent_id, branch_name)

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert args[0] == "git"
        assert args[1] == "worktree"
        assert args[2] == "add"
        assert args[3] == "-b"
        assert args[4] == branch_name
        assert args[5] == env_path
        assert args[6] == "HEAD"

@pytest.mark.asyncio
async def test_setup_isolation_failure(isolation_manager):
    agent_id = "test_agent_fail"

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"git error output")
    mock_process.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="Git worktree creation failed"):
            await isolation_manager.setup_isolation(agent_id)

        assert agent_id not in isolation_manager.active_contexts

@pytest.mark.asyncio
async def test_teardown_isolation_success(isolation_manager):
    agent_id = "test_agent_rm"
    env_path = "/tmp/test_magda_worktrees_v2/subagent_test_agent_rm_12345678"
    isolation_manager.active_contexts[agent_id] = env_path

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec, \
         patch("os.path.exists", return_value=True), \
         patch("shutil.rmtree") as mock_rmtree:

        await isolation_manager.teardown_isolation(agent_id)

        assert agent_id not in isolation_manager.active_contexts

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert args[0] == "git"
        assert args[1] == "worktree"
        assert args[2] == "remove"
        assert args[3] == "--force"
        assert args[4] == env_path

        mock_rmtree.assert_called_once_with(env_path)

@pytest.mark.asyncio
async def test_teardown_isolation_not_found(isolation_manager):
    agent_id = "test_agent_not_found"

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        await isolation_manager.teardown_isolation(agent_id)
        mock_exec.assert_not_called()

@pytest.mark.asyncio
async def test_teardown_isolation_git_failure(isolation_manager, caplog):
    agent_id = "test_agent_fail_rm"
    env_path = "/tmp/test_magda_worktrees_v2/subagent_test_agent_fail_rm_12345678"
    isolation_manager.active_contexts[agent_id] = env_path

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"git remove error")
    mock_process.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=mock_process), \
         patch("os.path.exists", return_value=True), \
         patch("shutil.rmtree") as mock_rmtree:

        await isolation_manager.teardown_isolation(agent_id)

        assert agent_id not in isolation_manager.active_contexts
        assert "Failed to remove isolated environment" in caplog.text
        # Cleanup should still be attempted
        mock_rmtree.assert_called_once_with(env_path)
