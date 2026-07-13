import pytest
import os
import sqlite3
from magda_agent.safety.acs_persistence import ACSPersistence
from magda_agent.safety.acs_guard_v6 import ACSGuardV6, SecurityViolationError

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_acs.db"
    return str(db_path)

def test_acs_persistence_logging(temp_db):
    persistence = ACSPersistence(db_path=temp_db)
    context = {"tool": "test_tool", "action": "read", "kwargs": {"path": "test.txt"}}

    persistence.log_checkpoint(1, "passed", "Input validation passed", context)
    persistence.log_checkpoint(2, "failed", "Unauthorized intent", context)

    logs = persistence.get_logs()
    assert len(logs) == 2

    # log 1: id, timestamp, checkpoint_id, status, reason, workflow_context
    assert logs[0][2] == 1
    assert logs[0][3] == "passed"
    assert "Input validation passed" in logs[0][4]
    assert "test_tool" in logs[0][5]

    assert logs[1][2] == 2
    assert logs[1][3] == "failed"
    assert "Unauthorized intent" in logs[1][4]

def test_acs_guard_integration(temp_db):
    persistence = ACSPersistence(db_path=temp_db)
    guard = ACSGuardV6(persistence=persistence)

    workflow_data = {
        "action": "read",
        "tool": "safe_tool",
        "kwargs": {"param": "value"},
        "current_state": "idle",
        "next_state": "executing"
    }

    # Mocking policy_layer to always allow
    guard.policy_layer.evaluate = lambda tool, **kwargs: (True, "Allowed by mock")

    guard.intercept_action(workflow_data)

    logs = persistence.get_logs()
    # Should have 5 logs, one for each checkpoint
    assert len(logs) == 5
    for i in range(5):
        assert logs[i][2] == i + 1
        assert logs[i][3] == "passed"

def test_acs_guard_failure_persistence(temp_db):
    persistence = ACSPersistence(db_path=temp_db)
    guard = ACSGuardV6(persistence=persistence)

    workflow_data = {
        "action": "unauthorized_action",
        "tool": "safe_tool",
        "kwargs": {"param": "value"}
    }

    with pytest.raises(SecurityViolationError):
        guard.intercept_action(workflow_data)

    logs = persistence.get_logs()
    # Checkpoint 1 should pass, Checkpoint 2 should fail
    assert len(logs) == 2
    assert logs[0][2] == 1
    assert logs[0][3] == "passed"
    assert logs[1][2] == 2
    assert logs[1][3] == "failed"
