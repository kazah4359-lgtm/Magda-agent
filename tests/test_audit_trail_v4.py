import pytest
import asyncio
import os
from magda_agent.safety.audit_trail_v4 import AuditTrailV4

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_audit_trail_v4.db")

@pytest.fixture
def audit_trail(db_path):
    return AuditTrailV4(db_path=db_path)

def test_audit_trail_v4_sync_interception(audit_trail):
    @audit_trail.intercept(tool_name="test_sync_tool")
    def sync_tool(x, y):
        return x + y

    result = sync_tool(2, 3)
    assert result == 5

    logs = audit_trail.get_logs()
    assert len(logs) == 1

    log = logs[0]
    assert log["tool_name"] == "test_sync_tool"
    assert log["args"] == [2, 3]
    assert log["result"] == 5
    assert log["duration"] >= 0

@pytest.mark.asyncio
async def test_audit_trail_v4_async_interception(audit_trail):
    @audit_trail.intercept(tool_name="test_async_tool")
    async def async_tool(data, password="secret_password"):
        await asyncio.sleep(0.01)
        return {"status": "ok", "data": data}

    result = await async_tool("some_data", password="super_secret")
    assert result["status"] == "ok"

    logs = audit_trail.get_logs()
    assert len(logs) == 1

    log = logs[0]
    assert log["tool_name"] == "test_async_tool"
    assert log["args"] == ["some_data"]
    # Keyword args should be scrubbed
    assert log["kwargs"]["password"] == "***"
    assert log["result"] == {"status": "ok", "data": "some_data"}
    assert log["duration"] >= 0.01

def test_audit_trail_v4_sanitization(audit_trail):
    @audit_trail.intercept(tool_name="sanitize_tool")
    def sensitive_tool(api_key, nested_data, user_auth_token):
        return {"private_key": "my_private_key", "public": "public_data"}

    sensitive_tool(
        api_key="12345",
        nested_data={"secret_code": "42", "normal_info": "hello"},
        user_auth_token="jwt123"
    )

    logs = audit_trail.get_logs()
    assert len(logs) == 1
    log = logs[0]

    # Assert scrubbing of inputs
    assert log["kwargs"]["api_key"] == "***"
    assert log["kwargs"]["user_auth_token"] == "***"

    # Assert nested dict scrubbing
    assert log["kwargs"]["nested_data"]["secret_code"] == "***"
    assert log["kwargs"]["nested_data"]["normal_info"] == "hello"

    # Assert scrubbing of output results
    assert log["result"]["private_key"] == "***"
    assert log["result"]["public"] == "public_data"

def test_audit_trail_v4_exception_logging(audit_trail):
    @audit_trail.intercept()
    def error_tool():
        raise ValueError("Something went wrong")

    with pytest.raises(ValueError):
        error_tool()

    logs = audit_trail.get_logs()
    assert len(logs) == 1
    assert logs[0]["tool_name"] == "error_tool"
    assert "Error: Something went wrong" in str(logs[0]["result"])
