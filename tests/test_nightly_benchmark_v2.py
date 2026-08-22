import pytest
import asyncio
from unittest.mock import MagicMock

from magda_agent.scheduler.cron_scheduler_v2 import CronSchedulerV2
from magda_agent.evaluation.nightly_benchmark_v2 import NightlyBenchmarkV2


@pytest.fixture
def mock_scheduler():
    scheduler = MagicMock(spec=CronSchedulerV2)
    return scheduler


def test_nightly_benchmark_registration(mock_scheduler):
    """Test that the benchmark correctly registers its task with the scheduler."""
    benchmark = NightlyBenchmarkV2(scheduler=mock_scheduler, cron_expr="0 2 * * *")
    benchmark.register()

    mock_scheduler.add_task.assert_called_once_with(
        name="nightly_benchmark_v2",
        cron_expr="0 2 * * *",
        func=benchmark.run_benchmark
    )


@pytest.mark.asyncio
async def test_nightly_benchmark_execution(mock_scheduler):
    """Test that the benchmark executes correctly and records mock results."""
    benchmark = NightlyBenchmarkV2(scheduler=mock_scheduler)

    # Initially results should be None
    assert benchmark.get_latest_results() is None

    # Run benchmark
    await benchmark.run_benchmark()

    # Results should be recorded
    results = benchmark.get_latest_results()
    assert results is not None
    assert results["status"] == "passed"
    assert results["swe_bench_score"] == 0.82
    assert results["agent_bench_score"] == 0.88
    assert results["regressions_detected"] is False
