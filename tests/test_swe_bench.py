import pytest
from unittest.mock import AsyncMock, patch

from magda_agent.evals.swe_bench import SWEBenchEvaluator

@pytest.fixture
def mock_llm():
    with patch("magda_agent.evals.swe_bench.LLMClient") as mock:
        yield mock.return_value

@pytest.fixture
def mock_tracker():
    with patch("magda_agent.evals.swe_bench.QualityTracker") as mock:
        yield mock.return_value

@pytest.fixture
def mock_consciousness():
    return AsyncMock()

@pytest.mark.asyncio
async def test_swe_bench_evaluator_simulation(mock_llm, mock_tracker):
    # Test simulation mode (no consciousness)
    mock_llm.generate = AsyncMock(return_value="0.8")
    evaluator = SWEBenchEvaluator(db_path=":memory:")

    score = await evaluator.run_task(evaluator.tasks[0])

    assert score == 0.8
    mock_llm.generate.assert_called_once()

@pytest.mark.asyncio
async def test_swe_bench_evaluator_with_consciousness(mock_llm, mock_tracker, mock_consciousness):
    # Test execution with consciousness
    mock_consciousness.process_input.return_value = "I have fixed the issue and run the tests."
    mock_llm.generate = AsyncMock(return_value="0.9")

    evaluator = SWEBenchEvaluator(db_path=":memory:", consciousness=mock_consciousness)

    score = await evaluator.run_task(evaluator.tasks[0])

    assert score == 0.9
    mock_consciousness.process_input.assert_called_once_with(evaluator.tasks[0]["goal"])
    mock_llm.generate.assert_called_once()

@pytest.mark.asyncio
async def test_swe_bench_run_evaluation_suite(mock_llm, mock_tracker):
    # Test running the full suite
    mock_llm.generate = AsyncMock(return_value="0.7")
    evaluator = SWEBenchEvaluator(db_path=":memory:")

    result = await evaluator.run_evaluation_suite()

    assert pytest.approx(result["score"], 0.01) == 0.7
    assert result["metadata"]["tasks_run"] == 3
    assert len(result["metadata"]["scores"]) == 3
    mock_tracker.log_metric.assert_called_once_with("swe_bench_score", result["score"], result["metadata"])

@pytest.mark.asyncio
async def test_swe_bench_evaluator_error_handling(mock_llm, mock_tracker, mock_consciousness):
    # Test error handling when consciousness fails
    mock_consciousness.process_input.side_effect = Exception("System failure")

    evaluator = SWEBenchEvaluator(db_path=":memory:", consciousness=mock_consciousness)

    score = await evaluator.run_task(evaluator.tasks[0])

    assert score == 0.0
