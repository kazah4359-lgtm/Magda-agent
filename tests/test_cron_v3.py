import pytest
import asyncio
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch

from magda_agent.operations.cron_v3 import HermesCronSchedulerV3

@pytest.fixture
def scheduler():
    # Provide a simple unique memory db
    return HermesCronSchedulerV3(db_path=":memory:")

@pytest.mark.asyncio
async def test_scheduler_persists_job(scheduler: HermesCronSchedulerV3):
    """
    Test that scheduling a job writes it to the SQLite database.
    """
    async def dummy_task():
        pass

    scheduler.schedule("* * * * *", dummy_task)

    conn = scheduler._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT name, cron_expr FROM jobs WHERE name = 'dummy_task'")
    row = cursor.fetchone()

    assert row is not None
    assert row[0] == "dummy_task"
    assert row[1] == "* * * * *"

@pytest.mark.asyncio
async def test_scheduler_handles_timezone_correctness(scheduler: HermesCronSchedulerV3):
    """
    Test that the scheduler uses timezone aware datetimes and handles correctness.
    """
    now_utc = datetime.now(timezone.utc)

    assert scheduler._get_now().tzinfo == timezone.utc

@pytest.mark.asyncio
async def test_daily_reports_execution(scheduler: HermesCronSchedulerV3):
    """
    Test a mock daily reports execution.
    """
    mock_task = AsyncMock(return_value="report generated")
    mock_callback = AsyncMock()

    scheduler.result_callback = mock_callback

    # 0 0 * * * is midnight daily
    scheduler.schedule("0 0 * * *", mock_task, name="daily_reports")

    # Manually trigger execution
    job = {
        "name": "daily_reports",
        "cron_expr": "0 0 * * *",
        "next_run": datetime.now(timezone.utc)
    }
    await scheduler._execute_job(job)

    mock_task.assert_called_once()
    mock_callback.assert_called_once_with("report generated")

    conn = scheduler._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT last_run, next_run FROM jobs WHERE name = 'daily_reports'")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] is not None
    assert row[1] is not None

@pytest.mark.asyncio
async def test_scheduler_decorator(scheduler: HermesCronSchedulerV3):
    """
    Test that the scheduler decorator successfully registers a task.
    """
    @scheduler.task("*/5 * * * *", name="decorated_task")
    async def my_task():
        return "done"

    conn = scheduler._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT name, cron_expr FROM jobs WHERE name = 'decorated_task'")
    row = cursor.fetchone()

    assert row is not None
    assert row[0] == "decorated_task"
    assert row[1] == "*/5 * * * *"

@pytest.mark.asyncio
async def test_scheduler_start_stop(scheduler: HermesCronSchedulerV3):
    """
    Test the start and stop operations of the scheduler loop.
    """
    await scheduler.start()
    assert scheduler._running is True
    assert scheduler._task is not None
    assert not scheduler._task.done()

    await scheduler.stop()
    assert scheduler._running is False
    assert scheduler._task.done()
