import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone
from magda_agent.operations.cron_v3 import HermesCronSchedulerV3
from magda_agent.operations.nightly_backup_v2 import NightlyBackupManagerV2

import pytest_asyncio

@pytest_asyncio.fixture
async def memory_scheduler():
    scheduler = HermesCronSchedulerV3(db_path=":memory:")
    yield scheduler
    await scheduler.stop()

@pytest.mark.asyncio
async def test_nightly_backup_registration(memory_scheduler):
    manager = NightlyBackupManagerV2(scheduler=memory_scheduler)
    mock_backup = AsyncMock()

    manager.register_nightly_backup("db_backup", mock_backup, cron_expr="0 2 * * *")

    # Check if the backup was registered
    assert "db_backup" in memory_scheduler._func_registry

    # Check if the job was stored in the in-memory db
    conn = memory_scheduler._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT name, cron_expr FROM jobs WHERE name = 'db_backup'")
    row = cursor.fetchone()

    assert row is not None
    assert row[0] == "db_backup"
    assert row[1] == "0 2 * * *"

@pytest.mark.asyncio
async def test_run_backup_now(memory_scheduler):
    manager = NightlyBackupManagerV2(scheduler=memory_scheduler)
    mock_backup = AsyncMock()

    manager.register_nightly_backup("db_backup", mock_backup)

    await manager.run_backup_now("db_backup")

    mock_backup.assert_awaited_once()

@pytest.mark.asyncio
async def test_start_stop_scheduler(memory_scheduler):
    manager = NightlyBackupManagerV2(scheduler=memory_scheduler)

    with patch.object(memory_scheduler, 'start', new_callable=AsyncMock) as mock_start:
        await manager.start()
        mock_start.assert_awaited_once()

    with patch.object(memory_scheduler, 'stop', new_callable=AsyncMock) as mock_stop:
        await manager.stop()
        mock_stop.assert_awaited_once()

@pytest.mark.asyncio
async def test_scheduled_backup_execution(memory_scheduler):
    manager = NightlyBackupManagerV2(scheduler=memory_scheduler)
    mock_backup = AsyncMock()

    # We will patch the scheduler's _get_now to return a time past the scheduled time
    # Default cron "0 2 * * *" means next run is 2 AM.
    # Let's mock time right before, schedule, then mock time after and step the loop manually.

    # Using 2026-06-01 01:00:00 as initial time
    start_time = datetime(2026, 6, 1, 1, 0, 0, tzinfo=timezone.utc)
    with patch.object(memory_scheduler, '_get_now', return_value=start_time):
        manager.register_nightly_backup("db_backup", mock_backup, cron_expr="0 2 * * *")

    # Fast forward to 2026-06-01 02:01:00
    future_time = datetime(2026, 6, 1, 2, 1, 0, tzinfo=timezone.utc)

    with patch.object(memory_scheduler, '_get_now', return_value=future_time):
        # We manually call _execute_job to simulate loop behavior since _loop has sleep.
        # Let's mock _loop one iteration. We can just query db for jobs to run.
        conn = memory_scheduler._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name, cron_expr, next_run FROM jobs")
        rows = cursor.fetchall()
        jobs_to_run = []
        for row in rows:
            name, cron_expr, next_run_str = row
            next_run = datetime.fromisoformat(next_run_str)
            if next_run.tzinfo is None:
                next_run = next_run.replace(tzinfo=timezone.utc)
            if future_time >= next_run:
                jobs_to_run.append({
                    "name": name,
                    "cron_expr": cron_expr,
                    "next_run": next_run
                })

        assert len(jobs_to_run) == 1
        assert jobs_to_run[0]["name"] == "db_backup"

        await memory_scheduler._execute_job(jobs_to_run[0])

    mock_backup.assert_awaited_once()
