import pytest
import os
import sqlite3
from magda_agent.safety.acs_persistence_v12 import ACSPersistenceV12

@pytest.fixture
def temp_db(tmp_path) -> str:
    db_path = tmp_path / "test_acs_v12.db"
    return str(db_path)

def test_acs_persistence_v12_logging(temp_db: str) -> None:
    """Tests basic logging of safety checkpoints in ACSPersistenceV12."""
    persistence = ACSPersistenceV12(db_path=temp_db)
    context = {"tool": "test_tool_v12", "action": "execute", "kwargs": {"param": 123}}

    persistence.log_checkpoint(1, "passed", "Input validation passed on param", context)
    persistence.log_checkpoint(2, "failed", "Unauthorized intent detected", context)

    logs = persistence.get_logs()
    assert len(logs) == 2

    # Verify ID, timestamp, checkpoint_id, status, reason, workflow_context
    # Entry 1
    assert logs[0][0] == 1  # ID
    assert logs[0][2] == 1  # checkpoint_id
    assert logs[0][3] == "passed"
    assert logs[0][4] == "Input validation passed on param"
    assert "test_tool_v12" in logs[0][5]

    # Entry 2
    assert logs[1][0] == 2
    assert logs[1][2] == 2
    assert logs[1][3] == "failed"
    assert logs[1][4] == "Unauthorized intent detected"
    assert "test_tool_v12" in logs[1][5]

def test_get_logs_filtering(temp_db: str) -> None:
    """Tests filtering of logs by checkpoint_id."""
    persistence = ACSPersistenceV12(db_path=temp_db)
    persistence.log_checkpoint(1, "passed", "Check 1")
    persistence.log_checkpoint(2, "passed", "Check 2")
    persistence.log_checkpoint(1, "failed", "Check 1 failed")

    logs_cp1 = persistence.get_logs(checkpoint_id=1)
    assert len(logs_cp1) == 2
    assert logs_cp1[0][4] == "Check 1"
    assert logs_cp1[1][4] == "Check 1 failed"

    logs_cp2 = persistence.get_logs(checkpoint_id=2)
    assert len(logs_cp2) == 1
    assert logs_cp2[0][4] == "Check 2"

def test_calculate_failure_rate(temp_db: str) -> None:
    """Tests the failure rate calculation logic."""
    persistence = ACSPersistenceV12(db_path=temp_db)

    # Empty DB failure rate should be 0.0
    assert persistence.calculate_failure_rate() == 0.0
    assert persistence.calculate_failure_rate(checkpoint_id=1) == 0.0

    # Populate checkpoint 1 logs: 1 passed, 1 failed (50% failure rate)
    persistence.log_checkpoint(1, "passed")
    persistence.log_checkpoint(1, "failed")

    # Populate checkpoint 2 logs: 1 passed (0% failure rate)
    persistence.log_checkpoint(2, "passed")

    # Calculate individual failure rates
    assert persistence.calculate_failure_rate(checkpoint_id=1) == 0.5
    assert persistence.calculate_failure_rate(checkpoint_id=2) == 0.0

    # Calculate overall failure rate (1 failed / 3 total = ~33.33%)
    assert pytest.approx(persistence.calculate_failure_rate(), 0.01) == 1.0 / 3.0

def test_count_total_runs(temp_db: str) -> None:
    """Tests counting of total checkpoint runs."""
    persistence = ACSPersistenceV12(db_path=temp_db)
    assert persistence.count_total_runs() == 0

    persistence.log_checkpoint(1, "passed")
    persistence.log_checkpoint(2, "failed")
    persistence.log_checkpoint(3, "passed")

    assert persistence.count_total_runs() == 3
