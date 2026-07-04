import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from magda_agent.operations.cron_reports import DailyReportManager
from magda_agent.operations.cron_v3 import HermesCronSchedulerV3

@pytest.fixture
def scheduler():
    return HermesCronSchedulerV3(db_path=":memory:")

@pytest.fixture
def report_manager(scheduler):
    return DailyReportManager(scheduler=scheduler)

@pytest.mark.asyncio
async def test_register_daily_report(report_manager):
    """
    Tests that a daily report function is registered properly with the scheduler.
    """
    mock_func = AsyncMock(return_value="Daily events aggregated.")

    report_manager.register_daily_report("test_report", mock_func, "0 9 * * *")

    conn = report_manager.scheduler._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT name, cron_expr FROM jobs WHERE name = 'test_report'")
    row = cursor.fetchone()

    assert row is not None
    assert row[0] == "test_report"
    assert row[1] == "0 9 * * *"

@pytest.mark.asyncio
async def test_generate_report_now(report_manager):
    """
    Tests that generate_report_now runs the correct registered function and returns its output.
    """
    mock_func = AsyncMock(return_value="Daily events aggregated.")

    report_manager.register_daily_report("test_report", mock_func)

    result = await report_manager.generate_report_now("test_report")
    assert result == "Daily events aggregated."
    mock_func.assert_called_once()

@pytest.mark.asyncio
async def test_generate_report_now_not_registered(report_manager):
    """
    Tests that generate_report_now handles unregistered report tasks gracefully.
    """
    result = await report_manager.generate_report_now("nonexistent_report")
    assert result is None

@pytest.mark.asyncio
async def test_generate_report_now_exception(report_manager):
    """
    Tests that generate_report_now handles exceptions raised by report functions gracefully.
    """
    mock_func = AsyncMock(side_effect=Exception("Simulation of report generation failure"))

    report_manager.register_daily_report("error_report", mock_func)

    result = await report_manager.generate_report_now("error_report")
    assert result is None
    mock_func.assert_called_once()

@pytest.mark.asyncio
async def test_report_manager_start_stop(report_manager):
    """
    Tests that the report manager correctly delegates start and stop to the scheduler.
    """
    with patch.object(report_manager.scheduler, 'start', new_callable=AsyncMock) as mock_start:
        await report_manager.start()
        mock_start.assert_called_once()

    with patch.object(report_manager.scheduler, 'stop', new_callable=AsyncMock) as mock_stop:
        await report_manager.stop()
        mock_stop.assert_called_once()
