import pytest
import os
import glob
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.scheduler.cron_reports_v3 import DailyReportManagerV3

@pytest.mark.asyncio
async def test_generate_report_success():
    # Mock dependencies
    mock_memory = MagicMock()
    mock_memory.get_all_events.return_value = [
        {"text": "Event 1"},
        {"text": "Event 2"}
    ]

    mock_procedural = MagicMock()
    mock_procedural.collection.get.return_value = {"ids": ["id1", "id2"]}

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "# Daily Report\nSummary of events."

    mock_scheduler = MagicMock()

    manager = DailyReportManagerV3(
        scheduler=mock_scheduler,
        memory=mock_memory,
        procedural=mock_procedural,
        llm=mock_llm
    )

    # Run report generation
    report = await manager.generate_report("Test Report")

    # Assertions
    assert report == "# Daily Report\nSummary of events."
    mock_llm.generate.assert_called_once()

    # Check if prompt contains expected stats
    prompt = mock_llm.generate.call_args[0][0]
    assert "Episodic Events: 2" in prompt
    assert "Procedural Learnings: 2" in prompt
    assert "- Event 1" in prompt
    assert "- Event 2" in prompt

    # Check if file is created
    report_files = glob.glob("daily_report_*.md")
    assert len(report_files) > 0

    # Cleanup
    for f in report_files:
        os.remove(f)

@pytest.mark.asyncio
async def test_generate_report_missing_config():
    manager = DailyReportManagerV3(memory=None, procedural=None, llm=None)
    report = await manager.generate_report("Fail Report")
    assert "not configured" in report

@pytest.mark.asyncio
async def test_generate_report_empty_events():
    mock_memory = MagicMock()
    mock_memory.get_all_events.return_value = []
    mock_procedural = MagicMock()
    mock_procedural.collection.get.return_value = {"ids": []}
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "No events summary"

    manager = DailyReportManagerV3(memory=mock_memory, procedural=mock_procedural, llm=mock_llm)
    report = await manager.generate_report("Empty Report")

    assert report == "No events summary"
    prompt = mock_llm.generate.call_args[0][0]
    assert "Episodic Events: 0" in prompt
    assert "Procedural Learnings: 0" in prompt

    # Cleanup
    report_files = glob.glob("daily_report_*.md")
    for f in report_files:
        os.remove(f)

@pytest.mark.asyncio
async def test_register_daily_report():
    mock_scheduler = MagicMock()
    manager = DailyReportManagerV3(scheduler=mock_scheduler)

    manager.register_daily_report("Daily Summary", "0 0 * * *")

    mock_scheduler.schedule.assert_called_once()
    args, kwargs = mock_scheduler.schedule.call_args
    assert args[0] == "0 0 * * *"
    assert kwargs['name'] == "Daily Summary"
