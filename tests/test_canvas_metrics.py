import pytest
from magda_agent.learning.openclaw_rl_metrics import OpenClawRLMetrics
from magda_agent.learning.canvas_metrics import RLCanvasMetricsExporter

def test_exporter_empty_metrics() -> None:
    """Test that the exporter handles empty metrics structures gracefully."""
    metrics = OpenClawRLMetrics()
    exporter = RLCanvasMetricsExporter(metrics)

    assert exporter.get_normalized_q_values() == {}
    assert exporter.get_moving_average_reward() == 0.0
    assert exporter.detect_reward_trend() == "insufficient_data"
    assert exporter.get_skill_activity_metrics() == {}

    payload = exporter.export_canvas_payload()
    assert payload["status"] == "active"
    assert payload["total_rewards_received"] == 0
    assert payload["trajectory_trend"] == "insufficient_data"
    assert payload["skills_coverage"] == {}
    assert payload["raw_rewards_trajectory"] == []

def test_exporter_normalized_q_values() -> None:
    """Test standard and edge cases of Q-values normalization."""
    metrics = OpenClawRLMetrics()
    exporter = RLCanvasMetricsExporter(metrics)

    # All identical values should map to 0.5
    metrics.update_q_value("skill_1", 1.5)
    metrics.update_q_value("skill_2", 1.5)
    normalized = exporter.get_normalized_q_values()
    assert normalized == {"skill_1": 0.5, "skill_2": 0.5}

    # Distinct values should scale properly [0.0, 1.0]
    metrics.update_q_value("skill_3", 0.0)
    metrics.update_q_value("skill_4", 3.0)
    distinct_normalized = exporter.get_normalized_q_values()
    assert distinct_normalized["skill_3"] == 0.0
    assert distinct_normalized["skill_4"] == 1.0
    assert distinct_normalized["skill_1"] == 0.5
    assert distinct_normalized["skill_2"] == 0.5

def test_exporter_moving_averages() -> None:
    """Test that the moving average considers only the defined window size."""
    metrics = OpenClawRLMetrics()
    exporter = RLCanvasMetricsExporter(metrics)

    metrics.add_reward("skill_a", 1.0)
    metrics.add_reward("skill_a", 2.0)
    metrics.add_reward("skill_a", 3.0)

    # Average of last 5 (which is 3 items actually present): (1+2+3)/3 = 2.0
    assert exporter.get_moving_average_reward(window_size=5) == 2.0

    # Average of last 2: (2+3)/2 = 2.5
    assert exporter.get_moving_average_reward(window_size=2) == 2.5

def test_exporter_reward_trends() -> None:
    """Test the detection of improving, declining, or stable trends."""
    metrics = OpenClawRLMetrics()
    exporter = RLCanvasMetricsExporter(metrics)

    # Not enough data initially
    assert exporter.detect_reward_trend() == "insufficient_data"

    # Improving: older is lower, newer is higher
    metrics.add_reward("skill_a", 0.1)
    metrics.add_reward("skill_a", 0.2)
    metrics.add_reward("skill_a", 0.8)
    metrics.add_reward("skill_a", 0.9)
    assert exporter.detect_reward_trend() == "improving"

    # Declining: older is higher, newer is lower
    metrics_declining = OpenClawRLMetrics()
    exporter_declining = RLCanvasMetricsExporter(metrics_declining)
    metrics_declining.add_reward("skill_a", 1.0)
    metrics_declining.add_reward("skill_a", 0.9)
    metrics_declining.add_reward("skill_a", 0.2)
    metrics_declining.add_reward("skill_a", 0.1)
    assert exporter_declining.detect_reward_trend() == "declining"

    # Stable: difference is very small
    metrics_stable = OpenClawRLMetrics()
    exporter_stable = RLCanvasMetricsExporter(metrics_stable)
    metrics_stable.add_reward("skill_a", 0.5)
    metrics_stable.add_reward("skill_a", 0.5)
    metrics_stable.add_reward("skill_a", 0.52)
    metrics_stable.add_reward("skill_a", 0.48)
    assert exporter_stable.detect_reward_trend() == "stable"

def test_exporter_skill_activity_metrics() -> None:
    """Test the aggregation of metrics per skill identifier."""
    metrics = OpenClawRLMetrics()
    exporter = RLCanvasMetricsExporter(metrics)

    metrics.add_reward("skill_alpha", 1.0)
    metrics.add_reward("skill_alpha", 2.0)
    metrics.add_reward("skill_beta", 4.0)

    stats = exporter.get_skill_activity_metrics()
    assert stats["skill_alpha"] == {"count": 2, "average_reward": 1.5}
    assert stats["skill_beta"] == {"count": 1, "average_reward": 4.0}

def test_exporter_canvas_payload_schema() -> None:
    """Test the structure and values of the complete Canvas payload."""
    metrics = OpenClawRLMetrics()
    exporter = RLCanvasMetricsExporter(metrics)

    metrics.update_q_value("skill_a", 1.0)
    metrics.update_q_value("skill_b", 2.0)
    metrics.add_reward("skill_a", 1.5)
    metrics.add_reward("skill_b", 2.5)

    payload = exporter.export_canvas_payload()
    assert payload["status"] == "active"
    assert payload["total_rewards_received"] == 2
    assert payload["global_average_q"] == 1.5
    assert payload["moving_average_reward_5"] == 2.0
    assert payload["trajectory_trend"] == "insufficient_data"
    assert "skill_a" in payload["skills_coverage"]
    assert "skill_b" in payload["skills_coverage"]

    skill_a_data = payload["skills_coverage"]["skill_a"]
    assert skill_a_data["q_value"] == 1.0
    assert skill_a_data["normalized_q_value"] == 0.0
    assert skill_a_data["activity"] == 1
    assert skill_a_data["avg_reward"] == 1.5
