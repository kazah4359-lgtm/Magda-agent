"""
Unit tests for WorktreeStateDifferV1 and worktree state diffing utility.
"""

import os
import subprocess
import pytest
from typing import Dict, List

from magda_agent.architecture.worktree_diffing_v1 import (
    WorktreeStateDifferV1,
    WorktreeDiffResult,
    WorktreeContaminationError,
    LeakType,
    IsolationLeak,
)


def test_clean_worktree_evaluation(tmp_path):
    wt_dir = tmp_path / "worktree_agent1"
    wt_dir.mkdir()

    # Create allowed file
    allowed_file = wt_dir / "src" / "module.py"
    allowed_file.parent.mkdir(parents=True)
    allowed_file.write_text("print('hello')", encoding="utf-8")

    differ = WorktreeStateDifferV1()
    result = differ.evaluate_worktree(
        worktree_path=str(wt_dir),
        agent_id="agent1",
        allowed_paths=["src/*"]
    )

    assert result.agent_id == "agent1"
    assert not result.is_contaminated
    assert len(result.leaks) == 0
    assert "src/module.py" in result.untracked_files or "src/module.py" in result.modified_files


def test_scope_violation_detection(tmp_path):
    wt_dir = tmp_path / "worktree_agent1"
    wt_dir.mkdir()

    # File within allowed scope
    allowed_file = wt_dir / "src" / "feature.py"
    allowed_file.parent.mkdir(parents=True)
    allowed_file.write_text("code", encoding="utf-8")

    # File outside allowed scope
    unauthorized_file = wt_dir / "billing" / "payments.py"
    unauthorized_file.parent.mkdir(parents=True)
    unauthorized_file.write_text("sensitive_billing()", encoding="utf-8")

    differ = WorktreeStateDifferV1()
    result = differ.evaluate_worktree(
        worktree_path=str(wt_dir),
        agent_id="agent1",
        allowed_paths=["src/*"]
    )

    assert result.is_contaminated
    scope_leaks = [leak for leak in result.leaks if leak.leak_type == LeakType.SCOPE_VIOLATION]
    assert len(scope_leaks) == 1
    assert scope_leaks[0].filepath == os.path.normpath("billing/payments.py")


def test_protected_path_detection(tmp_path):
    wt_dir = tmp_path / "worktree_agent1"
    wt_dir.mkdir()

    env_file = wt_dir / ".env"
    env_file.write_text("SECRET_KEY=12345", encoding="utf-8")

    secret_key_file = wt_dir / "private_secret.pem"
    secret_key_file.write_text("-----BEGIN RSA PRIVATE KEY-----", encoding="utf-8")

    differ = WorktreeStateDifferV1()
    result = differ.evaluate_worktree(
        worktree_path=str(wt_dir),
        agent_id="agent1"
    )

    assert result.is_contaminated
    protected_leaks = [leak for leak in result.leaks if leak.leak_type == LeakType.PROTECTED_PATH]
    assert len(protected_leaks) == 2
    filepaths = [leak.filepath for leak in protected_leaks]
    assert ".env" in filepaths
    assert "private_secret.pem" in filepaths


def test_symlink_escape_detection(tmp_path):
    wt_dir = tmp_path / "worktree_agent1"
    wt_dir.mkdir()

    outside_dir = tmp_path / "outside_system"
    outside_dir.mkdir()
    outside_target = outside_dir / "external_data.txt"
    outside_target.write_text("sensitive data", encoding="utf-8")

    symlink_file = wt_dir / "escaped_link"
    try:
        os.symlink(str(outside_target), str(symlink_file))
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on this filesystem or platform")

    differ = WorktreeStateDifferV1()
    result = differ.evaluate_worktree(
        worktree_path=str(wt_dir),
        agent_id="agent1"
    )

    assert result.is_contaminated
    symlink_leaks = [leak for leak in result.leaks if leak.leak_type == LeakType.EXTERNAL_SYMLINK]
    assert len(symlink_leaks) == 1
    assert symlink_leaks[0].filepath == "escaped_link"


def test_cross_worktree_overlap_detection(tmp_path):
    wt_agent1 = tmp_path / "wt1"
    wt_agent2 = tmp_path / "wt2"
    wt_agent1.mkdir()
    wt_agent2.mkdir()

    # Agent 1 modifies module_a.py and shared.py
    (wt_agent1 / "module_a.py").write_text("agent 1 code", encoding="utf-8")
    (wt_agent1 / "shared.py").write_text("agent 1 change", encoding="utf-8")

    # Agent 2 modifies module_b.py and shared.py
    (wt_agent2 / "module_b.py").write_text("agent 2 code", encoding="utf-8")
    (wt_agent2 / "shared.py").write_text("agent 2 change", encoding="utf-8")

    worktree_map = {
        "agent1": str(wt_agent1),
        "agent2": str(wt_agent2),
    }

    differ = WorktreeStateDifferV1()
    team_results = differ.evaluate_team_worktrees(worktree_map=worktree_map)

    assert team_results["agent1"].is_contaminated
    assert team_results["agent2"].is_contaminated

    assert "shared.py" in team_results["agent1"].overlapping_files
    assert "shared.py" in team_results["agent2"].overlapping_files

    overlap_leaks_1 = [leak for leak in team_results["agent1"].leaks if leak.leak_type == LeakType.CROSS_WORKTREE_OVERLAP]
    overlap_leaks_2 = [leak for leak in team_results["agent2"].leaks if leak.leak_type == LeakType.CROSS_WORKTREE_OVERLAP]

    assert len(overlap_leaks_1) == 1
    assert len(overlap_leaks_2) == 1
    assert overlap_leaks_1[0].filepath == "shared.py"


def test_validate_pre_merge_error(tmp_path):
    wt_dir = tmp_path / "worktree_agent1"
    wt_dir.mkdir()

    (wt_dir / ".env").write_text("DB_PASS=secret", encoding="utf-8")

    differ = WorktreeStateDifferV1()
    with pytest.raises(WorktreeContaminationError) as exc_info:
        differ.validate_pre_merge(worktree_path=str(wt_dir), agent_id="agent1")

    assert "Worktree merge validation failed for agent 'agent1'" in str(exc_info.value)
    assert ".env" in str(exc_info.value)


def test_git_status_porcelain_parsing(monkeypatch, tmp_path):
    wt_dir = tmp_path / "mock_git_wt"
    wt_dir.mkdir()

    porcelain_output = (
        " M src/modified.py\n"
        "?? src/new_file.py\n"
        " D src/deleted.py\n"
    )

    class MockCompletedProcess:
        returncode = 0
        stdout = porcelain_output
        stderr = ""

    def mock_run(*args, **kwargs):
        return MockCompletedProcess()

    monkeypatch.setattr(subprocess, "run", mock_run)

    differ = WorktreeStateDifferV1()
    changes = differ.get_worktree_changes(str(wt_dir))

    assert changes["modified"] == ["src/modified.py"]
    assert changes["untracked"] == ["src/new_file.py"]
    assert changes["deleted"] == ["src/deleted.py"]
