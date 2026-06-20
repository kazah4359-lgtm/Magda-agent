import pytest
import time
from magda_agent.safety.audit_trail import AuditTrail

def test_audit_trail_log_call() -> None:
    """Test that log_call correctly appends an entry to the trail."""
    trail_obj = AuditTrail()
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

def test_audit_trail_query() -> None:
    """Test that query correctly filters log entries."""
    trail_obj = AuditTrail()
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

def test_audit_trail_sanitize() -> None:
    """Test that sensitive data is correctly sanitized."""
    trail_obj = AuditTrail()
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

def test_audit_trail_bounded_capacity() -> None:
    """Test that the audit trail respects the maximum capacity."""
    trail_obj = AuditTrail(max_capacity=5)
    for i in range(10):
        trail_obj.log_call(f"tool_{i}", {}, "testing", "Success", 0.1)

    all_entries = trail_obj.get_all()
    assert len(all_entries) == 5
    assert all_entries[0]["tool_name"] == "tool_5"
    assert all_entries[-1]["tool_name"] == "tool_9"

def test_audit_trail_clear() -> None:
    """Test that clear empties the logs."""
    trail_obj = AuditTrail()
    trail_obj.log_call("test_tool", {}, "testing", "Success", 0.1)
    assert len(trail_obj.get_all()) == 1
    trail_obj.clear()
    assert len(trail_obj.get_all()) == 0
