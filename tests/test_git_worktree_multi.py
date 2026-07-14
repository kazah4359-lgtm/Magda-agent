import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from typing import List
from magda_agent.isolation.git_worktree_multi import GitWorktreeMultiManager

@pytest.mark.asyncio
async def test_execute_concurrently_success() -> None:
    """Test that GitWorktreeMultiManager executes tasks concurrently inside unique paths."""
    manager = GitWorktreeMultiManager(base_dir="/tmp/test_worktrees_multi")

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    paths_seen: List[str] = []

    async def sample_task(worktree_path: str) -> str:
        paths_seen.append(worktree_path)
        await asyncio.sleep(0.01)
        return f"Result from {worktree_path}"

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        results = await manager.execute_concurrently(
            tasks=[sample_task, sample_task, sample_task],
            branch_names=["b1", "b2", "b3"]
        )

        assert len(results) == 3
        assert len(set(paths_seen)) == 3
        for path in paths_seen:
            assert "test_worktrees_multi/worktree_" in path

        assert mock_exec.call_count == 6

@pytest.mark.asyncio
async def test_execute_concurrently_exception_handling() -> None:
    """Test that GitWorktreeMultiManager cleans up all worktrees even if one task raises an exception."""
    manager = GitWorktreeMultiManager(base_dir="/tmp/test_worktrees_multi")

    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"")
    mock_process.returncode = 0

    async def failing_task(worktree_path: str) -> str:
        await asyncio.sleep(0.01)
        raise ValueError("Task failed")

    async def succeeding_task(worktree_path: str) -> str:
        await asyncio.sleep(0.01)
        return "OK"

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        with pytest.raises(ValueError, match="Task failed"):
            await manager.execute_concurrently(
                tasks=[succeeding_task, failing_task]
            )

        assert mock_exec.call_count == 4
