import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.agents.isolation import IsolationManager

@pytest.mark.asyncio
async def test_create_isolated_environment_success():
    """Tests creating an isolated worktree environment successfully."""
    manager = IsolationManager(base_dir="/tmp/test_isolation")

    process_mock = MagicMock()
    process_mock.communicate = AsyncMock(return_value=(b"Success", b""))
    process_mock.returncode = 0

    with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = process_mock

        path = await manager.create_isolated_environment()

        assert path.startswith("/tmp/test_isolation/env_")
        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "git" in args
        assert "worktree" in args

@pytest.mark.asyncio
async def test_teardown_isolated_environment_success():
    """Tests tearing down an isolated worktree environment successfully."""
    manager = IsolationManager(base_dir="/tmp/test_isolation")

    process_mock = MagicMock()
    process_mock.communicate = AsyncMock(return_value=(b"Success", b""))
    process_mock.returncode = 0

    with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = process_mock

        await manager.teardown_isolated_environment("/tmp/test_isolation/env_abc")

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "remove" in args
        assert "/tmp/test_isolation/env_abc" in args
