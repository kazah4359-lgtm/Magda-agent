import pytest
import time
import os
from magda_agent.safety.audit_trail import AuditTrail

@pytest.fixture
def temp_db_path(tmp_path):
    """Fixture to provide a temporary database path."""
    return str(tmp_path / "test_audit_trail.db")

def test_audit_trail_log_call(temp_db_path) -> None:
    """Test that log_call correctly appends an entry to the trail."""
    trail_obj = AuditTrail(db_path=temp_db_path)
    trail_obj.log_call(
        tool_name="test_tool",
        kwargs={"arg1": "value1"},
        why="Because testing",
        result="Success",
        duration=0.5
    )

    all_entries = trail_obj.get_all()
    assert len(all_entries) == 1
    assert all_entries[0]["tool_name"] == "test_tool"
    assert all_entries[0]["kwargs"] == {"arg1": "value1"}
    assert all_entries[0]["why"] == "Because testing"
    assert all_entries[0]["result"] == "Success"
    assert all_entries[0]["duration"] == 0.5
    assert "timestamp" in all_entries[0]

def test_audit_trail_query(temp_db_path) -> None:
    """Test that query correctly filters log entries."""
    trail_obj = AuditTrail(db_path=temp_db_path)
    start_time = time.time()

    trail_obj.log_call("tool1", {}, "why1", "res1", 0.1)
    time.sleep(0.01)
    mid_time = time.time()
    time.sleep(0.01)
    trail_obj.log_call("tool2", {}, "why2", "res2", 0.2)

    assert len(trail_obj.query(tool_name="tool1")) == 1
    assert len(trail_obj.query(tool_name="tool2")) == 1
    assert len(trail_obj.query(tool_name="tool3")) == 0

    results = trail_obj.query(start_time=start_time, end_time=mid_time)
    assert len(results) == 1
    assert results[0]["tool_name"] == "tool1"

def test_audit_trail_sanitize(temp_db_path) -> None:
    """Test that sensitive data is correctly sanitized."""
    trail_obj = AuditTrail(db_path=temp_db_path)
    sensitive_kwargs = {
        "password": "my_secret_password",
        "api_key": "1234567890",
        "normal_arg": "normal_value",
        "nested": {
            "token": "secret_token",
            "auth_header": "Bearer xyz"
        },
        "list_of_secrets": [{"secret": "abc"}, {"normal": "def"}],
        "mixed_list": ["secret1", {"password": "pwd"}, "normal"]
    }

    trail_obj.log_call("test_tool", sensitive_kwargs, "testing", "Success", 0.1)

    all_entries = trail_obj.get_all()
    logged_kwargs = all_entries[0]["kwargs"]

    assert logged_kwargs["password"] == "***"
    assert logged_kwargs["api_key"] == "***"
    assert logged_kwargs["normal_arg"] == "normal_value"
    assert logged_kwargs["nested"]["token"] == "***"
    assert logged_kwargs["nested"]["auth_header"] == "***"
    assert logged_kwargs["list_of_secrets"][0]["secret"] == "***"
    assert logged_kwargs["list_of_secrets"][1]["normal"] == "def"

    # Note: mixed_list "secret1" won't be sanitized because it's not a dict key matching SENSITIVE_KEYS
    # but the dict inside mixed_list should be.
    # Actually my implementation of _sanitize on list only calls _sanitize(item).
    # If the item is a string, it returns copy.deepcopy(item).
    # If a dict key matches SENSITIVE_KEYS and value is a list, it sanitizes elements.

    # In my AuditTrail._sanitize:
    # elif isinstance(v, list):
    #     sanitized[k] = [
    #         self._sanitize(item) if isinstance(item, (dict, list)) else "***"
    #         for item in v
    #     ]
    # This only happens if k is in SENSITIVE_KEYS.

    # Let's check "api_key": ["secret1", "secret2"]
    sensitive_list = {"api_key": ["s1", "s2"]}
    trail_obj.log_call("test_tool2", sensitive_list, "testing", "Success", 0.1)
    entry2 = trail_obj.get_all()[1]
    assert entry2["kwargs"]["api_key"] == ["***", "***"]

def test_audit_trail_bounded_capacity(temp_db_path) -> None:
    """Test that the audit trail respects the maximum capacity."""
    trail_obj = AuditTrail(max_capacity=5, db_path=temp_db_path)
    for i in range(10):
        trail_obj.log_call(f"tool_{i}", {}, "testing", "Success", 0.1)

    all_entries = trail_obj.get_all()
    assert len(all_entries) == 5
    assert all_entries[0]["tool_name"] == "tool_5"
    assert all_entries[-1]["tool_name"] == "tool_9"

def test_audit_trail_clear(temp_db_path) -> None:
    """Test that clear empties the logs."""
    trail_obj = AuditTrail(db_path=temp_db_path)
    trail_obj.log_call("test_tool", {}, "testing", "Success", 0.1)
    assert len(trail_obj.get_all()) == 1

    # Also verify SQLite DB has 1 entry
    db_results = trail_obj.query_db()
    assert len(db_results) == 1

    trail_obj.clear()
    assert len(trail_obj.get_all()) == 0

    # Verify SQLite DB is empty
    db_results_after_clear = trail_obj.query_db()
    assert len(db_results_after_clear) == 0

def test_audit_trail_sqlite_logging(temp_db_path) -> None:
    """Test that log_call writes to the SQLite database correctly and query_db works."""
    trail_obj = AuditTrail(db_path=temp_db_path)

    start_time = time.time()
    trail_obj.log_call(
        tool_name="sqlite_tool",
        kwargs={"test_key": "test_value"},
        why="SQLite test",
        result={"status": "ok"},
        duration=0.5
    )
    time.sleep(0.01)
    mid_time = time.time()
    time.sleep(0.01)

    trail_obj.log_call(
        tool_name="sqlite_tool_2",
        kwargs={"test_key": "test_value2"},
        why="SQLite test 2",
        result="Success String",
        duration=0.3
    )

    db_entries = trail_obj.query_db()
    assert len(db_entries) == 2

    # Check serialization/deserialization
    assert db_entries[0]["tool_name"] == "sqlite_tool"
    assert db_entries[0]["kwargs"] == {"test_key": "test_value"}
    assert db_entries[0]["why"] == "SQLite test"
    assert db_entries[0]["result"] == {"status": "ok"}
    assert db_entries[0]["duration"] == 0.5

    assert db_entries[1]["tool_name"] == "sqlite_tool_2"
    assert db_entries[1]["result"] == "Success String"

    # Test query_db filters
    filtered_entries = trail_obj.query_db(tool_name="sqlite_tool_2")
    assert len(filtered_entries) == 1
    assert filtered_entries[0]["tool_name"] == "sqlite_tool_2"

    time_filtered_entries = trail_obj.query_db(start_time=start_time, end_time=mid_time)
    assert len(time_filtered_entries) == 1
    assert time_filtered_entries[0]["tool_name"] == "sqlite_tool"

def test_audit_trail_no_db() -> None:
    """Test that AuditTrail works normally when db_path is None."""
    trail_obj = AuditTrail(db_path=None)
    trail_obj.log_call("test_tool", {}, "testing", "Success", 0.1)

    assert len(trail_obj.get_all()) == 1
    assert trail_obj.query_db() == []

    trail_obj.clear()
    assert len(trail_obj.get_all()) == 0
