import pytest
import sqlite3
import os
from unittest.mock import MagicMock, AsyncMock, patch
from magda_agent.evaluation.agentbench import AgentBenchHarness

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_metrics.sqlite3"
    yield str(db_file)

@pytest.mark.asyncio
@patch("magda_agent.evaluation.agentbench.LLMClient")
async def test_run_evaluation_suite(mock_llm_client, temp_db):
    mock_instance = MagicMock()
    mock_instance.generate = AsyncMock(side_effect=["0.85", "0.70", "invalid"])
    mock_llm_client.return_value = mock_instance

    harness = AgentBenchHarness(db_path=temp_db)

    result = await harness.run_evaluation_suite("reasoning")
    assert result["score"] == 0.85
    assert result["metadata"]["suite"] == "reasoning"
    assert result["metadata"]["passed"] == 8

    result = await harness.run_evaluation_suite("web_navigation")
    assert result["score"] == 0.70

    # Test error handling when parsing response
    result = await harness.run_evaluation_suite("os_interaction")
    assert result["score"] == 0.50

    with pytest.raises(ValueError):
        await harness.run_evaluation_suite("unknown_suite")

@pytest.mark.asyncio
@patch("magda_agent.evaluation.agentbench.LLMClient")
async def test_trigger_evaluations_logs_metrics(mock_llm_client, temp_db):
    mock_instance = MagicMock()
    mock_instance.generate = AsyncMock(side_effect=["0.85", "0.70", "0.90"])
    mock_llm_client.return_value = mock_instance

    harness = AgentBenchHarness(db_path=temp_db)

    results = await harness.trigger_evaluations()
    assert len(results) == 3

    # Verify the metrics were logged to SQLite via QualityTracker
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT metric_name, value FROM metrics")
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 3
    metrics = {row[0]: row[1] for row in rows}
    assert metrics["agentbench_reasoning_score"] == 0.90 # the suites are queried in order: web_nav (0.85), os_inter (0.70), reasoning (0.90)
    assert metrics["agentbench_web_navigation_score"] == 0.85
    assert metrics["agentbench_os_interaction_score"] == 0.70
