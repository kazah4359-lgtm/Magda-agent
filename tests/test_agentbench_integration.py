import pytest
import sqlite3
import os
from unittest.mock import MagicMock, AsyncMock, patch
from magda_agent.evaluation.agentbench import AgentBenchHarness, daily_agentbench_eval

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_metrics_integration.sqlite3"
    yield str(db_file)

@pytest.mark.asyncio
@patch("magda_agent.evaluation.agentbench.LLMClient")
async def test_run_evaluation_suite_simulated(mock_llm_client, temp_db):
    mock_instance = MagicMock()
    # Mock scores for 2 tasks in one suite
    mock_instance.generate = AsyncMock(side_effect=["0.8", "0.9"])
    mock_llm_client.return_value = mock_instance

    # No consciousness provided, so it should simulate via LLM
    harness = AgentBenchHarness(db_path=temp_db)

    result = await harness.run_evaluation_suite("reasoning")
    assert pytest.approx(result["score"]) == 0.85 # (0.8 + 0.9) / 2
    assert result["metadata"]["suite"] == "reasoning"
    assert result["metadata"]["tasks_run"] == 2
    assert result["metadata"]["passed"] == 2 # both >= 0.7

@pytest.mark.asyncio
@patch("magda_agent.evaluation.agentbench.LLMClient")
async def test_run_evaluation_suite_with_consciousness(mock_llm_client, temp_db):
    mock_instance = MagicMock()
    # Mock LLM evaluation of agent responses
    mock_instance.generate = AsyncMock(side_effect=["1.0", "1.0"])
    mock_llm_client.return_value = mock_instance

    mock_consciousness = MagicMock()
    mock_consciousness.process_input = AsyncMock(return_value="Task completed successfully.")

    harness = AgentBenchHarness(db_path=temp_db, consciousness=mock_consciousness)

    result = await harness.run_evaluation_suite("coding")
    assert result["score"] == 1.0
    assert mock_consciousness.process_input.call_count == 2
    assert result["metadata"]["passed"] == 2

@pytest.mark.asyncio
@patch("magda_agent.evaluation.agentbench.AgentBenchHarness")
async def test_daily_agentbench_eval(mock_harness_class):
    mock_harness = MagicMock()
    mock_harness.trigger_evaluations = AsyncMock(return_value=[{"score": 0.8, "metadata": {}}])
    mock_harness_class.return_value = mock_harness

    results = await daily_agentbench_eval()
    assert len(results) == 1
    mock_harness.trigger_evaluations.assert_called_once()

@pytest.mark.asyncio
@patch("magda_agent.evaluation.agentbench.LLMClient")
async def test_trigger_evaluations_integration(mock_llm_client, temp_db):
    mock_instance = MagicMock()
    # 4 suites * 2 tasks each = 8 calls to generate (in simulation mode)
    mock_instance.generate = AsyncMock(return_value="0.75")
    mock_llm_client.return_value = mock_instance

    harness = AgentBenchHarness(db_path=temp_db)
    results = await harness.trigger_evaluations()

    assert len(results) == 4

    # Verify the metrics were logged to SQLite
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT metric_name, value FROM metrics")
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 4
    for row in rows:
        assert row[1] == 0.75
