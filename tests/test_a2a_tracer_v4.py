import asyncio
import pytest
from magda_agent.tracing.a2a_tracer_v4 import A2AAsyncTracer

@pytest.fixture
def tracer():
    return A2AAsyncTracer()

@pytest.mark.asyncio
async def test_a2a_tracer_isolation(tracer):
    """
    Test that concurrent async tasks maintain their own correlation IDs
    and correctly trace their actions without cross-contamination.
    """
    async def worker_1():
        # Set correlation ID for this task context
        cid = tracer.set_correlation_id("task_1_id")
        tracer.log_action("start_task_1")
        await asyncio.sleep(0.05)
        tracer.log_action("end_task_1", {"status": "success"})
        return cid

    async def worker_2():
        # Set a different correlation ID for this task context
        cid = tracer.set_correlation_id("task_2_id")
        tracer.log_action("start_task_2")
        await asyncio.sleep(0.05)
        tracer.log_action("end_task_2", {"status": "error"})
        return cid

    # Run tasks concurrently
    results = await asyncio.gather(worker_1(), worker_2())
    assert results == ["task_1_id", "task_2_id"]

    # Verify trace for task 1
    traces_1 = tracer.get_traces("task_1_id")
    assert len(traces_1) == 2
    assert traces_1[0]["action"] == "start_task_1"
    assert traces_1[1]["action"] == "end_task_1"
    assert traces_1[1]["payload"] == {"status": "success"}

    # Verify trace for task 2
    traces_2 = tracer.get_traces("task_2_id")
    assert len(traces_2) == 2
    assert traces_2[0]["action"] == "start_task_2"
    assert traces_2[1]["action"] == "end_task_2"
    assert traces_2[1]["payload"] == {"status": "error"}

@pytest.mark.asyncio
async def test_generate_correlation_id(tracer):
    """
    Test that a generated correlation ID is returned if none is provided.
    """
    cid = tracer.set_correlation_id()
    assert cid is not None
    assert tracer.get_correlation_id() == cid

    tracer.log_action("test_action")
    traces = tracer.get_traces(cid)
    assert len(traces) == 1
    assert traces[0]["action"] == "test_action"

@pytest.mark.asyncio
async def test_clear_correlation_id(tracer):
    """
    Test that correlation ID can be cleared from context.
    """
    tracer.set_correlation_id("test_id")
    assert tracer.get_correlation_id() == "test_id"
    tracer.clear_correlation_id()
    assert tracer.get_correlation_id() is None

    # Logging an action without a correlation ID should do nothing
    tracer.log_action("ignored_action")
    assert len(tracer.get_traces("test_id")) == 0
