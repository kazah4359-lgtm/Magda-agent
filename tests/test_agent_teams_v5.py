import asyncio
import os
import shutil
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magda_agent.architecture.agent_teams_v5 import (
    AgentWorktreeIsolationV5,
    AgentTeamManagerV5,
    GitWorktreeError,
    AgentEvaluatorTeamV5,
)

@pytest.fixture
def isolation_manager() -> AgentWorktreeIsolationV5:
    return AgentWorktreeIsolationV5(base_dir="/tmp/test_agent_teams_v5")

@pytest.fixture
def team_manager(isolation_manager: AgentWorktreeIsolationV5) -> AgentTeamManagerV5:
    return AgentTeamManagerV5(isolation_manager=isolation_manager)


def test_create_worktree_success(isolation_manager: AgentWorktreeIsolationV5) -> None:
    async def _run():
        agent_id = "agent_123"

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"output", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            path, env = await isolation_manager.create_worktree(agent_id)

            assert mock_exec.called
            assert agent_id in isolation_manager.active_worktrees
            assert path == isolation_manager.active_worktrees[agent_id]

            # Verify isolated env vars
            assert env["MAGDA_AGENT_ID"] == agent_id
            assert env["MAGDA_WORKTREE_PATH"] == path
            assert env["MAGDA_ISOLATED"] == "true"
            assert "MAGDA_WORKTREE_BRANCH" in env

    asyncio.run(_run())


def test_create_worktree_failure(isolation_manager: AgentWorktreeIsolationV5) -> None:
    async def _run():
        agent_id = "agent_fail"

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"git error")
        mock_process.returncode = 128

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(GitWorktreeError, match="Git worktree creation failed"):
                await isolation_manager.create_worktree(agent_id)

    asyncio.run(_run())


def test_synchronize_worktree_success(isolation_manager: AgentWorktreeIsolationV5) -> None:
    async def _run():
        agent_id = "agent_sync"
        dummy_path = "/tmp/test_agent_teams_v5/agent_sync_path"
        isolation_manager.active_worktrees[agent_id] = dummy_path
        isolation_manager.worktree_branches[agent_id] = "subagent/agent_sync_123"

        # Mock git subprocesses: add, status, commit, merge
        proc_add = AsyncMock()
        proc_add.communicate.return_value = (b"", b"")
        proc_add.returncode = 0

        proc_status = AsyncMock()
        proc_status.communicate.return_value = (b" M modified.py\n", b"")
        proc_status.returncode = 0

        proc_commit = AsyncMock()
        proc_commit.communicate.return_value = (b"[branch 123] commit", b"")
        proc_commit.returncode = 0

        proc_merge = AsyncMock()
        proc_merge.communicate.return_value = (b"Merge made by the 'ort' strategy.", b"")
        proc_merge.returncode = 0

        with patch("os.path.exists", return_value=True), \
             patch("asyncio.create_subprocess_exec", side_effect=[proc_add, proc_status, proc_commit, proc_merge]) as mock_exec:

            res = await isolation_manager.synchronize_worktree(agent_id, target_branch="main")

            assert res["success"] is True
            assert res["agent_id"] == agent_id
            assert res["committed"] is True
            assert res["synced"] is True
            assert mock_exec.call_count == 4

    asyncio.run(_run())


def test_synchronize_worktree_no_uncommitted_changes(isolation_manager: AgentWorktreeIsolationV5) -> None:
    async def _run():
        agent_id = "agent_no_changes"
        dummy_path = "/tmp/test_agent_teams_v5/agent_no_changes_path"
        isolation_manager.active_worktrees[agent_id] = dummy_path
        isolation_manager.worktree_branches[agent_id] = "subagent/agent_no_changes_123"

        proc_add = AsyncMock()
        proc_add.communicate.return_value = (b"", b"")
        proc_add.returncode = 0

        proc_status = AsyncMock()
        proc_status.communicate.return_value = (b"", b"")  # Clean worktree
        proc_status.returncode = 0

        proc_merge = AsyncMock()
        proc_merge.communicate.return_value = (b"Already up to date.", b"")
        proc_merge.returncode = 0

        with patch("os.path.exists", return_value=True), \
             patch("asyncio.create_subprocess_exec", side_effect=[proc_add, proc_status, proc_merge]) as mock_exec:

            res = await isolation_manager.synchronize_worktree(agent_id)

            assert res["success"] is True
            assert res["committed"] is False
            assert res["synced"] is True
            assert mock_exec.call_count == 3

    asyncio.run(_run())


def test_synchronize_worktree_error_on_merge(isolation_manager: AgentWorktreeIsolationV5) -> None:
    async def _run():
        agent_id = "agent_conflict"
        dummy_path = "/tmp/test_agent_teams_v5/agent_conflict_path"
        isolation_manager.active_worktrees[agent_id] = dummy_path
        isolation_manager.worktree_branches[agent_id] = "subagent/agent_conflict_123"

        proc_add = AsyncMock()
        proc_add.communicate.return_value = (b"", b"")
        proc_add.returncode = 0

        proc_status = AsyncMock()
        proc_status.communicate.return_value = (b"", b"")
        proc_status.returncode = 0

        proc_merge = AsyncMock()
        proc_merge.communicate.return_value = (b"", b"Automatic merge failed; fix conflicts and then commit the result.")
        proc_merge.returncode = 1

        with patch("os.path.exists", return_value=True), \
             patch("asyncio.create_subprocess_exec", side_effect=[proc_add, proc_status, proc_merge]):

            with pytest.raises(GitWorktreeError, match="Failed to synchronize worktree branch"):
                await isolation_manager.synchronize_worktree(agent_id)

    asyncio.run(_run())


def test_remove_worktree_success(isolation_manager: AgentWorktreeIsolationV5) -> None:
    async def _run():
        agent_id = "agent_remove"
        dummy_path = "/tmp/test_agent_teams_v5/agent_remove_path"
        isolation_manager.active_worktrees[agent_id] = dummy_path

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec, \
             patch("os.path.exists", return_value=False):

            await isolation_manager.remove_worktree(agent_id)

            mock_exec.assert_called_with("git", "worktree", "remove", "--force", dummy_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            assert agent_id not in isolation_manager.active_worktrees

    asyncio.run(_run())


def test_aggressive_cleanup_fallback(isolation_manager: AgentWorktreeIsolationV5) -> None:
    async def _run():
        agent_id = "agent_cleanup"
        dummy_path = "/tmp/test_agent_teams_v5/agent_cleanup_path"
        isolation_manager.active_worktrees[agent_id] = dummy_path

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"git error")
        mock_process.returncode = 128

        with patch("asyncio.create_subprocess_exec", return_value=mock_process), \
             patch("os.path.exists", side_effect=[True, False]), \
             patch("shutil.rmtree") as mock_rmtree:

            await isolation_manager.remove_worktree(agent_id)

            mock_rmtree.assert_called_once_with(dummy_path, ignore_errors=True)
            assert agent_id not in isolation_manager.active_worktrees

    asyncio.run(_run())


def test_team_manager_spawn_sync_and_disband(team_manager: AgentTeamManagerV5) -> None:
    async def _run():
        agent_id = "agent_manager"

        mock_path = "/tmp/mock_path"
        mock_env = {"MAGDA_AGENT_ID": agent_id}
        team_manager.isolation_manager.create_worktree = AsyncMock(return_value=(mock_path, mock_env))  # type: ignore
        team_manager.isolation_manager.synchronize_worktree = AsyncMock(return_value={"success": True, "agent_id": agent_id})  # type: ignore
        team_manager.isolation_manager.remove_worktree = AsyncMock()  # type: ignore

        path, env = await team_manager.spawn_agent(agent_id)
        assert path == mock_path
        assert env == mock_env
        assert agent_id in team_manager.agents

        # Synchronize
        sync_res = await team_manager.synchronize_subagent(agent_id)
        assert sync_res["success"] is True

        # Duplicate spawn error
        with pytest.raises(ValueError):
            await team_manager.spawn_agent(agent_id)

        # Disband with sync_before_disband=True
        await team_manager.disband_agent(agent_id, sync_before_disband=True)
        assert agent_id not in team_manager.agents
        assert team_manager.get_agent_env(agent_id) is None

    asyncio.run(_run())


def test_agent_evaluator_team_v5(team_manager: AgentTeamManagerV5) -> None:
    async def _run():
        evaluators = ["eval1", "eval2"]
        team = AgentEvaluatorTeamV5(evaluators=evaluators, team_manager=team_manager)

        with patch("magda_agent.llm_client.LLMClient.generate", new_callable=AsyncMock) as mock_generate, \
             patch.object(team_manager, "spawn_agent", new_callable=AsyncMock) as mock_spawn, \
             patch.object(team_manager, "disband_agent", new_callable=AsyncMock) as mock_disband:

            mock_generate.side_effect = [
                "PASSED: Code is good",
                "PASSED: Looks fine"
            ]

            mock_spawn.side_effect = [
                ("/tmp/eval1", {"MAGDA_WORKTREE_PATH": "/tmp/eval1"}),
                ("/tmp/eval2", {"MAGDA_WORKTREE_PATH": "/tmp/eval2"})
            ]

            result = await team.evaluate_code("def foo(): pass", {"context": "test"})

            assert result["passed"] is True
            assert result["overall_score"] == 100
            assert len(result["results"]) == 2

            assert mock_generate.call_count == 2
            assert mock_spawn.call_count == 2
            assert mock_disband.call_count == 2

    asyncio.run(_run())
