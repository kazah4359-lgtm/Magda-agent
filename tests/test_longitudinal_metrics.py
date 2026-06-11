import pytest
import sqlite3
import os
from magda_agent.evaluation.longitudinal_metrics import LongitudinalMetricsTracker
from magda_agent.evaluation.longitudinal import LongitudinalEvaluator
from magda_agent.metacognition.tracker import QualityTracker

@pytest.fixture
def test_db_path():
    path = "./test_longitudinal_metrics_db.sqlite3"
    yield path
    if os.path.exists(path):
        os.remove(path)

def test_record_and_retrieve_metric(test_db_path):
    tracker = LongitudinalMetricsTracker(db_path=test_db_path)

    # Record metrics
    tracker.record_metric("test_coverage", 85.5)
    tracker.record_metric("test_coverage", 86.2)
    tracker.record_metric("code_complexity", 10.5)

    # Retrieve metrics
    coverage_history = tracker.get_metrics_history("test_coverage", limit=10)
    complexity_history = tracker.get_metrics_history("code_complexity", limit=10)
    empty_history = tracker.get_metrics_history("non_existent_metric", limit=10)

    assert len(coverage_history) == 2
    # Since they are ordered by timestamp DESC, the last one added should be first
    assert coverage_history[0]["value"] == 86.2
    assert coverage_history[1]["value"] == 85.5

    assert len(complexity_history) == 1
    assert complexity_history[0]["value"] == 10.5

    assert len(empty_history) == 0

def create_mock_evaluator():
    evaluator = LongitudinalEvaluator(":memory:")
    # We need to artificially set timestamps to ensure correct DESC ordering
    # since fast execution might result in the same timestamp for all rows.
    conn = evaluator.tracker._get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO metrics (metric_name, value, timestamp) VALUES ('accuracy', 0.5, '2026-06-08 21:00:00')")
    cursor.execute("INSERT INTO metrics (metric_name, value, timestamp) VALUES ('accuracy', 0.6, '2026-06-08 21:01:00')")
    cursor.execute("INSERT INTO metrics (metric_name, value, timestamp) VALUES ('accuracy', 0.8, '2026-06-08 21:02:00')")
    conn.commit()
    return evaluator

def test_longitudinal_evaluator():
    evaluator = create_mock_evaluator()

    trend = evaluator.evaluate_trend("accuracy", limit=3)

    assert trend["metric_name"] == "accuracy"
    # The newest is 0.8, the oldest is 0.5. 0.8 > 0.5 => improving
    assert trend["trend"] == "improving"
    assert trend["data_points"] == 3
    assert round(trend["average"], 2) == 0.63
