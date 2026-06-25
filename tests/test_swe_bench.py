import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from magda_agent.evaluation.swe_bench import SWEBenchEvaluator

@pytest.fixture
def mock_llm_client():
    """Mocks the LLMClient for testing."""
    with patch('magda_agent.evaluation.swe_bench.LLMClient') as mock_llm:
        mock_instance = mock_llm.return_value
        mock_instance.chat_completion = AsyncMock(return_value="SUCCESS")
        yield mock_llm

@pytest.fixture
def mock_quality_tracker():
    """Mocks the QualityTracker for testing."""
    with patch('magda_agent.evaluation.swe_bench.QualityTracker') as mock_tracker:
        yield mock_tracker

@pytest.fixture
def mock_load_dataset():
    """Mocks the datasets.load_dataset function."""
    with patch('magda_agent.evaluation.swe_bench.load_dataset') as mock_load:
        mock_dataset = [
            {"problem_statement": "Fix issue 1", "repo": "test/repo"},
            {"problem_statement": "Fix issue 2", "repo": "test/repo"}
        ]
        mock_load.return_value = mock_dataset
        yield mock_load

@pytest.mark.asyncio
async def test_run_evaluation_suite(mock_llm_client, mock_quality_tracker, mock_load_dataset) -> None:
    """Tests the execution of an evaluation suite, ensuring score and metadata are returned correctly."""
    evaluator = SWEBenchEvaluator()
    result = await evaluator.run_evaluation_suite("swe_bench_verified")
    assert result["score"] == 1.0
    assert result["metadata"]["meets_baseline"] is True
    assert result["metadata"]["tasks_run"] == 2

@pytest.mark.asyncio
async def test_trigger_evaluations(mock_llm_client, mock_quality_tracker, mock_load_dataset) -> None:
    """Tests triggering evaluations for all suites and ensures scores are properly logged."""
    evaluator = SWEBenchEvaluator()
    results = await evaluator.trigger_evaluations()
    assert len(results) == 1
    assert results[0]["score"] == 1.0

def test_compare_with_baseline() -> None:
    """Tests the baseline comparison logic."""
    evaluator = SWEBenchEvaluator()
    assert evaluator.compare_with_baseline(0.85, "swe_bench_verified") is True
    assert evaluator.compare_with_baseline(0.75, "swe_bench_verified") is False
