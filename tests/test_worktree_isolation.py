import pytest
import os
import shutil
import asyncio
from unittest.mock import patch, MagicMock

from magda_agent.architecture.worktree_isolation import WorktreeIsolator

@pytest.mark.asyncio
async def test_worktree_isolation():
    isolator = WorktreeIsolator()

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        # Mock successful git worktree add
        mock_process = MagicMock()
        mock_process.returncode = 0

        future = asyncio.Future()
        future.set_result((b"ok", b""))
        mock_process.communicate.return_value = future
        mock_exec.return_value = mock_process

        path = await isolator.isolate_agent("test_agent", "test_branch")

        assert "test_agent" in isolator.active_worktrees
        assert path == isolator.active_worktrees["test_agent"]
        assert mock_exec.call_count == 1

        # Test clean up
        with patch("shutil.rmtree") as mock_rmtree, patch("os.path.exists", return_value=True):
            await isolator.cleanup_agent("test_agent")
            assert "test_agent" not in isolator.active_worktrees
            assert mock_exec.call_count == 2
            mock_rmtree.assert_called_once_with(path)
