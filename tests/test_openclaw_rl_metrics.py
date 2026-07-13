import pytest
from unittest.mock import MagicMock
from magda_agent.learning.openclaw_rl_metrics import OpenClawRLMetrics
from magda_agent.visualization.canvas_v3 import CanvasVisualizerV3

def test_openclaw_rl_metrics_updates():
    metrics = OpenClawRLMetrics()

    # Test Q-value update
    metrics.update_q_value("test_skill", 0.5)
    assert metrics.q_values["test_skill"] == 0.5

    # Test reward addition
    metrics.add_reward("test_skill", 1.0, user_id="user123")
    assert len(metrics.recent_rewards) == 1
    assert metrics.recent_rewards[0]["reward"] == 1.0
    assert metrics.recent_rewards[0]["skill_id"] == "test_skill"
    assert metrics.recent_rewards[0]["user_id"] == "user123"

def test_openclaw_rl_metrics_max_rewards():
    metrics = OpenClawRLMetrics()
    metrics.max_recent_rewards = 5

    for i in range(10):
        metrics.add_reward("skill", float(i))

    assert len(metrics.recent_rewards) == 5
    assert metrics.recent_rewards[-1]["reward"] == 9.0

def test_openclaw_rl_metrics_visualization_data():
    metrics = OpenClawRLMetrics()
    metrics.update_q_value("skill1", 0.8)
    metrics.update_q_value("skill2", 0.4)
    metrics.add_reward("skill1", 1.0)

    viz_data = metrics.get_visualization_data()
    assert viz_data["q_values"]["skill1"] == 0.8
    assert pytest.approx(viz_data["average_q"]) == 0.6
    assert viz_data["reward_count"] == 1
    assert viz_data["status"] == "active"

def test_canvas_visualizer_v3_with_rl_metrics():
    # Mock consciousness
    mock_consciousness = MagicMock()
    mock_consciousness.emotions = None
    mock_consciousness.mental_states = None
    mock_consciousness.memory = None
    mock_consciousness.skills = None
    mock_consciousness.planner = None
    mock_consciousness.hypothalamus = None

    # Add RL metrics to mock
    rl_metrics = OpenClawRLMetrics()
    rl_metrics.update_q_value("skill_a", 0.9)
    mock_consciousness.openclaw_rl_metrics = rl_metrics

    visualizer = CanvasVisualizerV3(mock_consciousness)
    state = visualizer.get_formatted_state()

    assert "rl_metrics" in state
    assert state["rl_metrics"]["q_values"]["skill_a"] == 0.9
    assert state["rl_metrics"]["status"] == "active"
