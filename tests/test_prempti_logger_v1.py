import pytest
from magda_agent.safety.audit_trail import AuditTrail
from magda_agent.safety.prempti_logger_v1 import PremptiToolLoggerV1
from magda_agent.safety.taint import mark_tainted


@pytest.fixture
def audit_trail():
    return AuditTrail(max_capacity=100, db_path=None)


@pytest.fixture
def logger(audit_trail):
    return PremptiToolLoggerV1(audit_trail=audit_trail)


def test_log_pre_execution_directly(logger, audit_trail):
    metadata = logger.log_pre_execution(
        tool_name="test_tool",
        kwargs={"param1": "value1", "param2": 123},
        why="manual test"
    )

    assert metadata["tool_name"] == "test_tool"
    assert metadata["status"] == "pre_execution"
    assert metadata["why"] == "manual test"

    logs = audit_trail.get_all()
    assert len(logs) == 1
    log = logs[0]
    assert log["tool_name"] == "test_tool"
    assert log["result"] == "pre_execution"
    assert log["kwargs"]["param1"] == "value1"
    assert log["kwargs"]["param2"] == 123
    assert log["kwargs"]["_tainted_boundary_crossover"] is False


def test_intercept_without_execution(logger, audit_trail):
    executed = False

    @logger.intercept(tool_name="no_exec_tool", why="skip exec", execute_tool=False)
    def dummy_func(x: int):
        nonlocal executed
        executed = True
        return x * 2

    res = dummy_func(10)
    assert res is None
    assert executed is False

    logs = audit_trail.get_all()
    assert len(logs) == 1
    log = logs[0]
    assert log["tool_name"] == "no_exec_tool"
    assert log["result"] == "pre_execution"
    assert log["why"] == "skip exec (pre-execution)"
    assert log["kwargs"]["x"] == 10


def test_intercept_sync_execution(logger, audit_trail):
    @logger.intercept(tool_name="sync_tool", why="normal execution")
    def add(a: int, b: int) -> int:
        return a + b

    res = add(3, 4)
    assert res == 7

    logs = audit_trail.get_all()
    assert len(logs) == 2  # 1 pre-execution, 1 execution
    assert logs[0]["result"] == "pre_execution"
    assert logs[1]["result"] == 7
    assert logs[1]["tool_name"] == "sync_tool"
    assert logs[1]["kwargs"]["a"] == 3
    assert logs[1]["kwargs"]["b"] == 4


def test_intercept_tainted_arguments_and_result(logger, audit_trail):
    @logger.intercept(tool_name="tainted_tool")
    def process(data: str) -> str:
        return mark_tainted(f"processed_{data}")

    res = process(mark_tainted("dirty_input"))

    logs = audit_trail.get_all()
    assert len(logs) == 2
    pre_log = logs[0]
    post_log = logs[1]

    assert pre_log["kwargs"]["_tainted_boundary_crossover"] is True
    assert post_log["kwargs"]["_tainted_boundary_crossover"] is True


@pytest.mark.asyncio
async def test_intercept_async_execution(logger, audit_trail):
    @logger.intercept(tool_name="async_tool")
    async def multiply_async(val: int):
        return val * 3

    res = await multiply_async(4)
    assert res == 12

    logs = audit_trail.get_all()
    assert len(logs) == 2
    assert logs[0]["result"] == "pre_execution"
    assert logs[1]["result"] == 12
    assert logs[1]["tool_name"] == "async_tool"


@pytest.mark.asyncio
async def test_intercept_async_without_execution(logger, audit_trail):
    executed = False

    @logger.intercept(tool_name="async_no_exec", execute_tool=False)
    async def async_dummy(val: int):
        nonlocal executed
        executed = True
        return val

    res = await async_dummy(100)
    assert res is None
    assert executed is False

    logs = audit_trail.get_all()
    assert len(logs) == 1
    assert logs[0]["result"] == "pre_execution"
    assert logs[0]["tool_name"] == "async_no_exec"
