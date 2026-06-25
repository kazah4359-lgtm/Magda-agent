import pytest
import asyncio
from magda_agent.safety.audit_trail import AuditTrail
from magda_agent.safety.audit import ToolInterceptor

@pytest.fixture
def audit_trail() -> AuditTrail:
    """Fixture for AuditTrail."""
    return AuditTrail(max_capacity=10)

@pytest.fixture
def interceptor(audit_trail: AuditTrail) -> ToolInterceptor:
    """Fixture for ToolInterceptor."""
    return ToolInterceptor(audit_trail)

def test_intercept_sync_success(interceptor: ToolInterceptor, audit_trail: AuditTrail) -> None:
    """Test successful synchronous interception."""
    @interceptor.intercept(tool_name="sync_tool", why="testing sync success")
    def sync_tool(x: int, y: int = 5) -> int:
        return x + y

    result = sync_tool(10)
    assert result == 15

    logs = audit_trail.query(tool_name="sync_tool")
    assert len(logs) == 1
    log = logs[0]
    assert log["kwargs"]["x"] == 10
    assert log["kwargs"]["y"] == 5
    assert log["why"] == "testing sync success"
    assert log["result"] == 15
    assert log["duration"] >= 0

def test_intercept_sync_exception(interceptor: ToolInterceptor, audit_trail: AuditTrail) -> None:
    """Test synchronous interception throwing exception."""
    @interceptor.intercept(tool_name="sync_tool_fail", why="testing sync fail")
    def sync_tool_fail(val: str) -> None:
        raise ValueError("Failed intentionally")

    with pytest.raises(ValueError):
        sync_tool_fail("hello")

    logs = audit_trail.query(tool_name="sync_tool_fail")
    assert len(logs) == 1
    log = logs[0]
    assert log["kwargs"]["val"] == "hello"
    assert "testing sync fail (failed)" in log["why"]
    assert log["result"] == "Failed intentionally"

@pytest.mark.asyncio
async def test_intercept_async_success(interceptor: ToolInterceptor, audit_trail: AuditTrail) -> None:
    """Test successful async interception."""
    @interceptor.intercept(tool_name="async_tool", why="testing async success")
    async def async_tool(a: str, b: str) -> str:
        await asyncio.sleep(0.01)
        return a + b

    result = await async_tool("foo", b="bar")
    assert result == "foobar"

    logs = audit_trail.query(tool_name="async_tool")
    assert len(logs) == 1
    log = logs[0]
    assert log["kwargs"]["a"] == "foo"
    assert log["kwargs"]["b"] == "bar"
    assert log["result"] == "foobar"
    assert log["duration"] > 0

@pytest.mark.asyncio
async def test_intercept_async_exception(interceptor: ToolInterceptor, audit_trail: AuditTrail) -> None:
    """Test async interception throwing exception."""
    @interceptor.intercept(tool_name="async_tool_fail", why="testing async fail")
    async def async_tool_fail() -> None:
        raise RuntimeError("Async error")

    with pytest.raises(RuntimeError):
        await async_tool_fail()

    logs = audit_trail.query(tool_name="async_tool_fail")
    assert len(logs) == 1
    log = logs[0]
    assert "testing async fail (failed)" in log["why"]
    assert log["result"] == "Async error"

def test_execute_sync_success(interceptor: ToolInterceptor, audit_trail: AuditTrail) -> None:
    """Test execute_sync method successfully."""
    def raw_tool(x: int) -> int:
        return x * 2

    result = interceptor.execute_sync(raw_tool, "raw_sync", "testing execute_sync", 20)
    assert result == 40

    logs = audit_trail.query(tool_name="raw_sync")
    assert len(logs) == 1
    assert logs[0]["kwargs"]["x"] == 20
    assert logs[0]["result"] == 40

@pytest.mark.asyncio
async def test_execute_async_success(interceptor: ToolInterceptor, audit_trail: AuditTrail) -> None:
    """Test execute_async method successfully."""
    async def raw_async_tool(y: int) -> int:
        return y * 3

    result = await interceptor.execute_async(raw_async_tool, "raw_async", "testing execute_async", 10)
    assert result == 30

    logs = audit_trail.query(tool_name="raw_async")
    assert len(logs) == 1
    assert logs[0]["kwargs"]["y"] == 10
    assert logs[0]["result"] == 30
