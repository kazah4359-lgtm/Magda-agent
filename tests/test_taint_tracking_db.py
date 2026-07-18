"""Tests for Taint Tracking DB."""
import sqlite3
import concurrent.futures
import pytest
from typing import Any, Dict
from magda_agent.safety.taint import MCPKernel, PolicyViolationError
from magda_agent.safety.taint_tracking_db import TaintTrackingDB, execute_with_tracking

def sample_tool(data: Any) -> str:
    """A simple tool for testing."""
    return f"Processed {data}"

def test_db_initialization_and_logging():
    """Test that the DB is initialized and can log items."""
    conn = sqlite3.connect(":memory:")
    db = TaintTrackingDB(db_path=":memory:")

    db.log_execution(
        tool_name="test_tool",
        inputs={"data": "test"},
        is_sensitive=False,
        violation_raised=False,
        error_message=None,
        conn=conn
    )

    results = db.query(conn=conn)
    assert len(results) == 1
    assert results[0]["tool_name"] == "test_tool"
    assert "test" in results[0]["inputs"]
    assert not results[0]["is_sensitive"]
    assert not results[0]["violation_raised"]

    conn.close()

def test_execute_with_tracking_success():
    """Test successful execution tracking."""
    conn = sqlite3.connect(":memory:")
    db = TaintTrackingDB(db_path=":memory:")
    kernel = MCPKernel()

    result = execute_with_tracking(
        kernel=kernel,
        db=db,
        tool_name="sample_tool",
        tool_func=sample_tool,
        inputs={"data": "safe"},
        is_sensitive=True,
        conn=conn
    )

    assert result == "Processed safe"

    logs = db.query(conn=conn)
    assert len(logs) == 1
    assert logs[0]["tool_name"] == "sample_tool"
    assert logs[0]["violation_raised"] is False
    assert logs[0]["is_sensitive"] is True

    conn.close()

def test_execute_with_tracking_violation():
    """Test violation execution tracking."""
    conn = sqlite3.connect(":memory:")
    db = TaintTrackingDB(db_path=":memory:")
    kernel = MCPKernel()

    tainted_input = kernel.tracker.taint("bad_data")

    with pytest.raises(PolicyViolationError):
        execute_with_tracking(
            kernel=kernel,
            db=db,
            tool_name="sample_tool_sensitive",
            tool_func=sample_tool,
            inputs={"data": tainted_input},
            is_sensitive=True,
            conn=conn
        )

    logs = db.query(conn=conn)
    assert len(logs) == 1
    assert logs[0]["tool_name"] == "sample_tool_sensitive"
    assert logs[0]["violation_raised"] is True
    assert logs[0]["is_sensitive"] is True
    assert "Tainted input" in logs[0]["error_message"]

    conn.close()

def test_execute_with_tracking_concurrent():
    """Test concurrent logging to SQLite."""
    import tempfile
    import os

    # We need a file DB for concurrent access testing, memory DBs don't share well across threads easily
    fd, temp_db_path = tempfile.mkstemp()
    os.close(fd)

    try:
        db = TaintTrackingDB(db_path=temp_db_path, timeout=10.0)
        kernel = MCPKernel()

        def run_tool(i):
            try:
                execute_with_tracking(
                    kernel=kernel,
                    db=db,
                    tool_name=f"tool_{i}",
                    tool_func=sample_tool,
                    inputs={"data": f"input_{i}"},
                    is_sensitive=False
                )
            except Exception:
                pass

        num_threads = 20
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(run_tool, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        logs = db.query()
        assert len(logs) == num_threads

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
