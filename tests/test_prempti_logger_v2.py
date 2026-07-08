import pytest
import asyncio
from typing import Any
from magda_agent.safety.prempti_logger_v2 import PremptiAuditLoggerV2

def test_sync_interception() -> None:
    """Test that a synchronous function can be intercepted and its arguments sanitized."""
    logger = PremptiAuditLoggerV2()

    @logger.intercept(tool_name="test_sync", why="testing sync")
    def sync_tool(param1: str, password: str) -> str:
        return f"{param1} and secret done"

    result = sync_tool("public", password="super_secret_password")
    assert result == "public and secret done"

    logs = logger.get_all()
    assert len(logs) == 1
    log_entry = logs[0]
    assert log_entry["tool_name"] == "test_sync"
    assert log_entry["why"] == "testing sync"
    assert log_entry["result"] == "public and secret done"
    assert "kwargs" in log_entry
    assert log_entry["kwargs"]["param1"] == "public"
    assert log_entry["kwargs"]["password"] == "***"
    assert log_entry["duration"] >= 0.0

@pytest.mark.asyncio
async def test_async_interception() -> None:
    """Test that an asynchronous function can be intercepted and its arguments sanitized."""
    logger = PremptiAuditLoggerV2()

    @logger.intercept(why="testing async")
    async def async_tool(data: dict) -> str:
        await asyncio.sleep(0.01)
        return "async done"

    result = await async_tool({"public_data": "123", "api_key": "secret456"})
    assert result == "async done"

    logs = logger.get_all()
    assert len(logs) == 1
    log_entry = logs[0]
    assert log_entry["tool_name"] == "async_tool"
    assert log_entry["why"] == "testing async"
    assert log_entry["result"] == "async done"

    kwargs = log_entry["kwargs"]
    assert "data" in kwargs
    assert kwargs["data"]["public_data"] == "123"
    assert kwargs["data"]["api_key"] == "***"
    assert log_entry["duration"] >= 0.005 # Sometimes it's slightly faster or sleep yields early

def test_sanitization_rules() -> None:
    """Test the sanitization logic accurately redacts sensitive data and allows non-sensitive."""
    logger = PremptiAuditLoggerV2()

    data = {
        "normal": "value",
        "PASSWORD": "123",
        "my_Secret_key": "456",
        "nested": {
            "token": "789",
            "safe": "safe_val"
        },
        "list_of_secrets": [{"api_key": "abc"}, {"normal": "def"}],
        "list_of_primitives_with_secret_key": ["a", "b", "c"],
        "token_list": ["secret1", "secret2"],
        "author": "john doe",  # Should not be redacted even though 'auth' is a sensitive keyword
        "environment": "prod", # Should not be redacted even though 'env' is a sensitive keyword
        "keynote": "hello"     # Should not be redacted even though 'key' is a sensitive keyword
    }

    sanitized = logger._sanitize(data)

    assert sanitized["normal"] == "value"
    assert sanitized["PASSWORD"] == "***"
    assert sanitized["my_Secret_key"] == "***"

    assert sanitized["nested"]["safe"] == "safe_val"
    assert sanitized["nested"]["token"] == "***"

    assert sanitized["list_of_secrets"] == "***"

    assert sanitized["token_list"] == "***"

    assert sanitized["author"] == "john doe"
    assert sanitized["environment"] == "prod"
    assert sanitized["keynote"] == "hello"

def test_exception_logging() -> None:
    """Test that exceptions during tool execution are correctly caught and logged."""
    logger = PremptiAuditLoggerV2()

    @logger.intercept(tool_name="error_tool", why="testing error")
    def error_tool(token: str) -> None:
        raise ValueError("Something went wrong")

    with pytest.raises(ValueError, match="Something went wrong"):
        error_tool("my_secret_token")

    logs = logger.get_all()
    assert len(logs) == 1
    log_entry = logs[0]
    assert log_entry["tool_name"] == "error_tool"
    assert log_entry["why"] == "testing error (failed)"
    assert log_entry["result"] == "Something went wrong"
    assert log_entry["kwargs"]["token"] == "***"

class UnpicklableObject:
    def __init__(self, value: str):
        self.value = value

    def __deepcopy__(self, memo: dict) -> Any:
        raise TypeError("Cannot deepcopy this object")

def test_unpicklable_object_sanitization() -> None:
    """Test that the logger does not crash when attempting to serialize unpicklable objects."""
    logger = PremptiAuditLoggerV2()

    obj = UnpicklableObject("some state")

    @logger.intercept(tool_name="unpicklable_tool", why="testing unpicklable")
    def unpicklable_tool(param1: Any) -> Any:
        return param1

    result = unpicklable_tool(obj)

    logs = logger.get_all()
    assert len(logs) == 1
    assert "UnpicklableObject" in logs[0]["kwargs"]["param1"]
    assert "UnpicklableObject" in logs[0]["result"]

def test_circular_reference() -> None:
    """Test that circular references do not cause infinite recursion in sanitization."""
    logger = PremptiAuditLoggerV2()

    data: Dict[str, Any] = {"hello": "world"}
    data["self_ref"] = data

    sanitized = logger._sanitize(data)
    assert sanitized["hello"] == "world"
    assert sanitized["self_ref"] == "<circular reference>"
