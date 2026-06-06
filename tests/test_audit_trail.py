import pytest
import time
from magda_agent.tracing.audit import AuditLogger

def test_audit_logger_log_call() -> None:
    """Test that log_call correctly appends an entry to the trail."""
    logger = AuditLogger()
    logger.log_call(
        tool_name="test_tool",
        kwargs={"arg1": "value1"},
        why="Because testing",
        result="Success",
        duration=0.5
    )

    trail = logger.get_all()
    assert len(trail) == 1
    assert trail[0]["tool_name"] == "test_tool"
    assert trail[0]["kwargs"] == {"arg1": "value1"}
    assert trail[0]["why"] == "Because testing"
    assert trail[0]["result"] == "Success"
    assert trail[0]["duration"] == 0.5
    assert "timestamp" in trail[0]

def test_audit_logger_query() -> None:
    """Test that query correctly filters log entries."""
    logger = AuditLogger()
    start_time = time.time()

    logger.log_call("tool1", {}, "why1", "res1", 0.1)
    time.sleep(0.01)
    mid_time = time.time()
    time.sleep(0.01)
    logger.log_call("tool2", {}, "why2", "res2", 0.2)

    end_time = time.time()

    assert len(logger.query(tool_name="tool1")) == 1
    assert len(logger.query(tool_name="tool2")) == 1
    assert len(logger.query(tool_name="tool3")) == 0

    assert len(logger.query(start_time=start_time, end_time=mid_time)) == 1
    assert logger.query(start_time=start_time, end_time=mid_time)[0]["tool_name"] == "tool1"

def test_audit_logger_sanitize() -> None:
    """Test that sensitive data is correctly sanitized."""
    logger = AuditLogger()

    sensitive_kwargs = {
        "password": "my_secret_password",
        "api_key": "1234567890",
        "normal_arg": "normal_value",
        "nested": {
            "token": "secret_token",
            "auth_header": "Bearer xyz"
        },
        "list_of_secrets": [{"secret_key": "abc"}, {"normal": "def"}]
    }

    logger.log_call("test_tool", sensitive_kwargs, "testing", "Success", 0.1)

    trail = logger.get_all()
    logged_kwargs = trail[0]["kwargs"]

    assert logged_kwargs["password"] == "***"
    assert logged_kwargs["api_key"] == "***"
    assert logged_kwargs["normal_arg"] == "normal_value"
    assert logged_kwargs["nested"]["token"] == "***"
    assert logged_kwargs["nested"]["auth_header"] == "***"
    assert logged_kwargs["list_of_secrets"][0]["secret_key"] == "***"
    assert logged_kwargs["list_of_secrets"][1]["normal"] == "def"

def test_audit_logger_clear() -> None:
    """Test that clear correctly empties the log."""
    logger = AuditLogger()
    logger.log_call("tool1", {}, "why1", "res1", 0.1)
    assert len(logger.get_all()) == 1
    logger.clear()
    assert len(logger.get_all()) == 0
