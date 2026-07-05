import pytest
import pytest_asyncio
import asyncio
from unittest.mock import MagicMock, AsyncMock

from magda_agent.operations.cron_v3 import HermesCronSchedulerV3
from magda_agent.operations.cron_reports_v3 import DailyReportManagerV3
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.llm_client import LLMClient

@pytest_asyncio.fixture
async def scheduler():
    sched = HermesCronSchedulerV3()
    yield sched
    await sched.stop()

@pytest.fixture
def mock_memory():
    memory = MagicMock(spec=EpisodicMemory)
    memory.get_all_events.return_value = [
        {"text": "Event 1 happened"},
        {"text": "Event 2 happened"}
    ]
    return memory

@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.generate = AsyncMock(return_value="This is a generated daily report.")
    return llm

@pytest.mark.asyncio
async def test_register_daily_report(scheduler, mock_memory, mock_llm):
    manager = DailyReportManagerV3(scheduler=scheduler, memory=mock_memory, llm=mock_llm)
    manager.register_daily_report("daily_update")

    assert "daily_update" in scheduler._func_registry

@pytest.mark.asyncio
async def test_generate_aggregated_report(scheduler, mock_memory, mock_llm):
    manager = DailyReportManagerV3(scheduler=scheduler, memory=mock_memory, llm=mock_llm)

    report = await manager.generate_aggregated_report("daily_update")

    assert report == "This is a generated daily report."
    mock_memory.get_all_events.assert_called_once_with(limit=100)
    mock_llm.generate.assert_called_once()

    call_args = mock_llm.generate.call_args[0][0]
    assert "Event 1 happened" in call_args
    assert "Event 2 happened" in call_args
    assert "daily_update" in call_args

@pytest.mark.asyncio
async def test_generate_report_now(scheduler, mock_memory, mock_llm):
    manager = DailyReportManagerV3(scheduler=scheduler, memory=mock_memory, llm=mock_llm)
    manager.register_daily_report("daily_update")

    report = await manager.generate_report_now("daily_update")
    assert report == "This is a generated daily report."

@pytest.mark.asyncio
async def test_generate_report_no_events(scheduler, mock_memory, mock_llm):
    mock_memory.get_all_events.return_value = []
    manager = DailyReportManagerV3(scheduler=scheduler, memory=mock_memory, llm=mock_llm)

    report = await manager.generate_aggregated_report("daily_update")
    assert report == "Report daily_update: No recent events to report."
    mock_llm.generate.assert_not_called()

@pytest.mark.asyncio
async def test_generate_report_missing_deps(scheduler):
    manager = DailyReportManagerV3(scheduler=scheduler, memory=None, llm=None)

    report = await manager.generate_aggregated_report("daily_update")
    assert report == "Error: Memory or LLM not configured."
