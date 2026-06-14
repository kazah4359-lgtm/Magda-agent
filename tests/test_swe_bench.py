import pytest
from unittest.mock import AsyncMock, patch
from magda_agent.evaluation.swe_bench import SWEBenchEvaluator

@pytest.fixture
def mock_llm_client():
    with patch('magda_agent.evaluation.swe_bench.LLMClient') as mock_llm:
        mock_instance = mock_llm.return_value
        mock_instance.chat_completion = AsyncMock(return_value="0.85")
        yield mock_llm

@pytest.fixture
def mock_quality_tracker():
    with patch('magda_agent.evaluation.swe_bench.QualityTracker') as mock_tracker:
        yield mock_tracker

@pytest.mark.asyncio
async def test_run_evaluation_suite(mock_llm_client, mock_quality_tracker):
    evaluator = SWEBenchEvaluator()
    result = await evaluator.run_evaluation_suite("swe_bench_verified")
    assert result["score"] == 0.85
    assert result["metadata"]["meets_baseline"] is True

@pytest.mark.asyncio
async def test_trigger_evaluations(mock_llm_client, mock_quality_tracker):
    evaluator = SWEBenchEvaluator()
    results = await evaluator.trigger_evaluations()
    assert len(results) == 1
    assert results[0]["score"] == 0.85

def test_compare_with_baseline():
    evaluator = SWEBenchEvaluator()
    assert evaluator.compare_with_baseline(0.85, "swe_bench_verified") is True
    assert evaluator.compare_with_baseline(0.75, "swe_bench_verified") is False
