import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

from magda_agent.scheduler.cron_reports import DailyReportScheduler
from magda_agent.scheduler.cron import CronScheduler

@pytest.fixture
def scheduler():
    return DailyReportScheduler()

@pytest.mark.asyncio
async def test_register_daily_report(scheduler):
    mock_report = AsyncMock()

    scheduler.register_daily_report("daily_summary", mock_report)

    assert "daily_summary" in scheduler._registered_reports

    # Check underlying CronScheduler
    assert len(scheduler.scheduler.jobs) == 1
    job = scheduler.scheduler.jobs[0]
    assert job["name"] == "daily_summary"
    assert job["cron_expr"] == "0 9 * * *"

@pytest.mark.asyncio
async def test_register_nightly_backup(scheduler):
    mock_backup = AsyncMock()

    scheduler.register_nightly_backup("db_backup", mock_backup)

    assert "db_backup" in scheduler._registered_reports

    # Check underlying CronScheduler
    assert len(scheduler.scheduler.jobs) == 1
    job = scheduler.scheduler.jobs[0]
    assert job["name"] == "db_backup"
    assert job["cron_expr"] == "0 2 * * *"

@pytest.mark.asyncio
async def test_scheduler_triggers_reports_on_time():
    ops_scheduler = CronScheduler()
    scheduler = DailyReportScheduler(scheduler=ops_scheduler)

    mock_report = AsyncMock(return_value="report_generated")
    mock_backup = AsyncMock(return_value="backup_completed")

    scheduler.register_daily_report("daily_summary", mock_report, cron_expr="0 9 * * *")
    scheduler.register_nightly_backup("db_backup", mock_backup, cron_expr="0 2 * * *")

    now = datetime(2026, 6, 1, 1, 59, 59) # Right before backup

    with patch.object(ops_scheduler, '_get_now') as mock_get_now:
        mock_get_now.return_value = now

        # Manually force the iterator next run for testing
        from croniter import croniter
        for job in ops_scheduler.jobs:
            job["iterator"] = croniter(job["cron_expr"], now)
            job["next_run"] = job["iterator"].get_next(datetime)

        ops_scheduler._running = True

        # Advance time to backup run
        mock_get_now.return_value = datetime(2026, 6, 1, 2, 0, 0)

        for job in ops_scheduler.jobs:
            if mock_get_now() >= job["next_run"]:
                await ops_scheduler._execute_job(job)
                job["next_run"] = job["iterator"].get_next(datetime)

        mock_backup.assert_called_once()
        mock_report.assert_not_called()

        # Advance time to daily report run
        mock_get_now.return_value = datetime(2026, 6, 1, 9, 0, 0)

        for job in ops_scheduler.jobs:
            if mock_get_now() >= job["next_run"]:
                await ops_scheduler._execute_job(job)
                job["next_run"] = job["iterator"].get_next(datetime)

        mock_report.assert_called_once()
        mock_backup.assert_called_once() # still 1

@pytest.mark.asyncio
async def test_scheduler_start_stop(scheduler):
    await scheduler.start()
    assert scheduler.scheduler._running is True
    assert scheduler.scheduler._task is not None
    assert not scheduler.scheduler._task.done()

    await scheduler.stop()
    assert scheduler.scheduler._running is False
    assert scheduler.scheduler._task.done()
# Added for PR diff visibility
