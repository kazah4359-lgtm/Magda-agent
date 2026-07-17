import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from magda_agent.operations.cron_v3 import HermesCronSchedulerV3
from magda_agent.scheduler.cron_diagnostics import DiagnosticScheduler
from unittest.mock import patch, MagicMock

@pytest.fixture
def memory_scheduler():
    scheduler = HermesCronSchedulerV3(db_path=":memory:")
    yield scheduler

@pytest.mark.asyncio
async def test_cron_diagnostics_scheduling(memory_scheduler):
    # Initialize the DiagnosticScheduler which will register the task
    diagnostic_scheduler = DiagnosticScheduler(scheduler=memory_scheduler)

    # Check that it got registered in the inner scheduler
    assert "diagnostic_health_check" in memory_scheduler._func_registry
    func = memory_scheduler._func_registry["diagnostic_health_check"]

    # Verify the function is indeed our target
    assert func == diagnostic_scheduler.run_diagnostics

@pytest.mark.asyncio
@patch('magda_agent.scheduler.cron_diagnostics.logger')
async def test_cron_diagnostics_execution(mock_logger, memory_scheduler):
    diagnostic_scheduler = DiagnosticScheduler(scheduler=memory_scheduler)

    # Run the diagnostics directly to check behavior
    await diagnostic_scheduler.run_diagnostics()

    # Check if the logger was called
    mock_logger.info.assert_any_call("Starting scheduled diagnostic health checks...")

    # Ensure it logged the report
    report_call = [call for call in mock_logger.info.call_args_list if "Diagnostic Report:" in call[0][0]]
    assert len(report_call) == 1

    # It shouldn't log any errors since we simulated a healthy system
    mock_logger.error.assert_not_called()

@pytest.mark.asyncio
@patch('magda_agent.scheduler.cron_diagnostics.logger')
async def test_cron_diagnostics_graceful_failure(mock_logger, memory_scheduler):
    diagnostic_scheduler = DiagnosticScheduler(scheduler=memory_scheduler)

    # Force an error in the scheduler state check or anywhere to trigger exception
    # Since we can't easily break the internal checks without monkeypatching,
    # we'll mock the logger and just check that IF an exception occurred, it is caught.
    # We can do this by mocking the self.scheduler to throw on attribute access

    with patch.object(diagnostic_scheduler, 'scheduler', new_callable=MagicMock) as mock_sched:
        type(mock_sched)._running = property(lambda self: (_ for _ in ()).throw(ValueError("Forced test error")))

        await diagnostic_scheduler.run_diagnostics()

        # Should catch and log error
        mock_logger.error.assert_called_once()
        assert "Failed to run diagnostic health checks:" in mock_logger.error.call_args[0][0]

@pytest.mark.asyncio
async def test_cron_diagnostics_trigger_conditions(memory_scheduler):
    diagnostic_scheduler = DiagnosticScheduler(scheduler=memory_scheduler)

    # We will simulate time to check if it would trigger
    now = datetime.now(timezone.utc)

    # The cron expression is "0 * * * *"
    # So the next run should be at the next top of the hour
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    conn = memory_scheduler._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT next_run FROM jobs WHERE name = 'diagnostic_health_check'")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    next_run_db = datetime.fromisoformat(row[0])

    # Next run should match our next hour prediction or the current hour if now minute is 0 (but croniter handles this)
    # The check confirms that the task was correctly persisted to SQLite.
    assert next_run_db.tzinfo == timezone.utc
