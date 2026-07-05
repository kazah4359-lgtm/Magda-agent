import pytest
import sqlite3
import os
from magda_agent.metacognition.longitudinal_tracker import LongitudinalTracker


@pytest.fixture
def temp_db(tmp_path):
    """Fixture to provide a temporary database path for tests."""
    db_path = tmp_path / "test_metrics.db"
    yield str(db_path)


def test_tracker_initialization(temp_db):
    """Test that the tracker initializes the database correctly."""
    tracker = LongitudinalTracker(db_path=temp_db)

    assert os.path.exists(temp_db)

    # Verify the table exists
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pr_metrics'")
        assert cursor.fetchone() is not None


def test_record_pr_outcome(temp_db):
    """Test recording PR outcomes."""
    tracker = LongitudinalTracker(db_path=temp_db)

    tracker.record_pr_outcome("PR-1", "success", 0)
    tracker.record_pr_outcome("PR-2", "failure", 5)

    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT pr_id, outcome, test_failures FROM pr_metrics ORDER BY id")
        rows = cursor.fetchall()

        assert len(rows) == 2
        assert rows[0] == ("PR-1", "success", 0)
        assert rows[1] == ("PR-2", "failure", 5)


def test_get_summary(temp_db):
    """Test the summary method."""
    tracker = LongitudinalTracker(db_path=temp_db)

    # Empty DB
    summary = tracker.get_summary()
    assert summary["total_prs"] == 0
    assert summary["success_rate"] == 0.0
    assert summary["total_test_failures"] == 0

    # Add data
    tracker.record_pr_outcome("PR-1", "success", 0)
    tracker.record_pr_outcome("PR-2", "failure", 3)
    tracker.record_pr_outcome("PR-3", "success", 0)

    summary = tracker.get_summary()
    assert summary["total_prs"] == 3
    assert summary["success_rate"] == 66.67
    assert summary["total_test_failures"] == 3


def test_bounded_storage(temp_db):
    """Test that the tracker respects the max_entries limit."""
    tracker = LongitudinalTracker(db_path=temp_db, max_entries=2)

    tracker.record_pr_outcome("PR-1", "success", 0)
    tracker.record_pr_outcome("PR-2", "failure", 3)
    tracker.record_pr_outcome("PR-3", "success", 0)

    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT pr_id FROM pr_metrics ORDER BY id")
        rows = cursor.fetchall()

        assert len(rows) == 2
        assert rows[0][0] == "PR-2"
        assert rows[1][0] == "PR-3"

    summary = tracker.get_summary()
    assert summary["total_prs"] == 2
    assert summary["success_rate"] == 50.0
    assert summary["total_test_failures"] == 3
