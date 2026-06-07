import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from magda_agent.scheduler.cron import CronScheduler

@pytest.fixture
def scheduler():
    return CronScheduler()

@pytest.mark.asyncio
async def test_scheduler_accepts_cron_expression(scheduler):
    async def dummy_task():
        pass

    # Valid cron
    scheduler.schedule("* * * * *", dummy_task)
    assert len(scheduler.jobs) == 1
    assert scheduler.jobs[0]["cron_expr"] == "* * * * *"

    # Invalid cron
    with pytest.raises(ValueError, match="Invalid cron expression"):
        scheduler.schedule("invalid cron", dummy_task)

@pytest.mark.asyncio
async def test_scheduler_executes_task_on_schedule():
    mock_task = AsyncMock(return_value="result")
    mock_callback = AsyncMock()

    scheduler = CronScheduler(result_callback=mock_callback)

    now = datetime(2026, 6, 1, 12, 0, 0)

    # Mock _get_now to control time
    with patch.object(scheduler, '_get_now') as mock_get_now:
        mock_get_now.return_value = now

        # Schedule for every minute
        scheduler.schedule("* * * * *", mock_task)
        assert len(scheduler.jobs) == 1
        job = scheduler.jobs[0]
        assert job["next_run"] == datetime(2026, 6, 1, 12, 1, 0)

        # Start scheduler loop briefly to verify it skips running early
        scheduler._running = True

        # Fast forward time to just before next run
        mock_get_now.return_value = datetime(2026, 6, 1, 12, 0, 59)
        # Call the loop logic once manually without sleeping
        for j in scheduler.jobs:
            if mock_get_now() >= j["next_run"]:
                await scheduler._execute_job(j)
                j["next_run"] = j["iterator"].get_next(datetime)

        mock_task.assert_not_called()

        # Fast forward time to next run
        mock_get_now.return_value = datetime(2026, 6, 1, 12, 1, 0)
        for j in scheduler.jobs:
            if mock_get_now() >= j["next_run"]:
                await scheduler._execute_job(j)
                j["next_run"] = j["iterator"].get_next(datetime)

        mock_task.assert_called_once()
        mock_callback.assert_called_once_with("result")

        # Verify next run is calculated correctly
        assert job["next_run"] == datetime(2026, 6, 1, 12, 2, 0)

@pytest.mark.asyncio
async def test_scheduler_start_stop(scheduler):
    await scheduler.start()
    assert scheduler._running is True
    assert scheduler._task is not None
    assert not scheduler._task.done()

    await scheduler.stop()
    assert scheduler._running is False
    assert scheduler._task.done()

@pytest.mark.asyncio
async def test_scheduler_list_and_clear_jobs(scheduler):
    async def dummy_task():
        pass
    scheduler.schedule("* * * * *", dummy_task)
    assert len(scheduler.list_jobs()) == 1
    scheduler.clear_jobs()
    assert len(scheduler.list_jobs()) == 0
