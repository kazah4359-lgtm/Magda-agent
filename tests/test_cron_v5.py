import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from magda_agent.scheduler.cron_v5 import CronScheduler

@pytest.fixture
def scheduler():
    return CronScheduler()

@pytest.mark.asyncio
async def test_schedule_valid_cron(scheduler):
    async def dummy_task():
        pass

    scheduler.schedule("0 0 * * *", dummy_task, name="daily_task")
    assert len(scheduler.jobs) == 1
    assert scheduler.jobs[0]["name"] == "daily_task"
    assert scheduler.jobs[0]["cron_expr"] == "0 0 * * *"

@pytest.mark.asyncio
async def test_schedule_invalid_cron(scheduler):
    async def dummy_task():
        pass

    with pytest.raises(ValueError, match="Invalid cron expression"):
        scheduler.schedule("invalid cron", dummy_task)

@pytest.mark.asyncio
async def test_start_stop(scheduler):
    await scheduler.start()
    assert scheduler._running is True
    assert scheduler._task is not None
    assert not scheduler._task.done()

    await scheduler.stop()
    assert scheduler._running is False
    assert scheduler._task.done() or scheduler._task.cancelled()

@pytest.mark.asyncio
async def test_execute_job(scheduler):
    mock_func = MagicMock()
    async def dummy_task(*args, **kwargs):
        mock_func(*args, **kwargs)
        return "result"

    callback = MagicMock()
    async def dummy_callback(res):
        callback(res)

    scheduler.result_callback = dummy_callback

    # Schedule a task
    scheduler.schedule("* * * * *", dummy_task, "test_job", "value1", kwarg1="value2")
    job = scheduler.jobs[0]

    # Execute it directly
    await scheduler._execute_job(job)

    mock_func.assert_called_once_with("value1", kwarg1="value2")
    callback.assert_called_once_with("result")

@pytest.mark.asyncio
async def test_loop_execution():
    scheduler = CronScheduler()

    call_count = 0
    async def dummy_task():
        nonlocal call_count
        call_count += 1

    # Mock time inside the scheduler BEFORE scheduling
    # Need to keep it mutable for _get_now
    time_dict = {"current": datetime(2023, 1, 1, 12, 0, 0)}
    def mock_get_now():
        return time_dict["current"]
    scheduler._get_now = mock_get_now

    # Schedule to run every minute
    scheduler.schedule("* * * * *", dummy_task, name="test_job")
    job = scheduler.jobs[0]

    # Next run should be 12:01:00
    assert job["next_run"] == datetime(2023, 1, 1, 12, 1, 0)

    # Store original sleep
    original_sleep = asyncio.sleep

    # Start scheduler loop, but mock sleep to yield to event loop quickly
    async def mock_sleep(*args):
        await original_sleep(0.001)

    with patch("asyncio.sleep", mock_sleep):
        await scheduler.start()

        # Advance time past the next run
        time_dict["current"] = datetime(2023, 1, 1, 12, 1, 1)

        # Let the event loop run long enough for the loop to catch the new time
        await original_sleep(0.05)

        await scheduler.stop()

    assert call_count >= 1
    # Next run should now be 12:02:00
    assert job["next_run"] == datetime(2023, 1, 1, 12, 2, 0)
