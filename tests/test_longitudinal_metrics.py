import pytest
import sqlite3
from magda_agent.evaluation.longitudinal import LongitudinalEvaluator
from magda_agent.metacognition.tracker import QualityTracker

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
