import pytest
from unittest.mock import MagicMock
from magda_agent.learning.rl_metrics_v4 import RLMetricsTrackerV4
from magda_agent.metacognition.tracker import QualityTracker

def test_log_reward() -> None:
    """Tests that logging a reward correctly calls the underlying quality tracker."""
    mock_tracker = MagicMock(spec=QualityTracker)
    rl_tracker = RLMetricsTrackerV4(mock_tracker)

    rl_tracker.log_reward(1.0, "search_skill", user_id="user123")

    mock_tracker.log_metric.assert_called_once_with(
        "rl_reward_v4", 1.0,
        {"skill_name": "search_skill", "user_id": "user123", "type": "reward"}
    )

def test_log_weight_delta() -> None:
    """Tests that logging a weight delta correctly calls the underlying quality tracker."""
    mock_tracker = MagicMock(spec=QualityTracker)
    rl_tracker = RLMetricsTrackerV4(mock_tracker)

    rl_tracker.log_weight_delta(0.1, "search_skill", user_id="user123")

    mock_tracker.log_metric.assert_called_once_with(
        "rl_weight_delta_v4", 0.1,
        {"skill_name": "search_skill", "user_id": "user123", "type": "weight_delta"}
    )

def test_get_reward_trend() -> None:
    """Tests that getting the reward trend correctly filters metrics by skill name."""
    mock_tracker = MagicMock(spec=QualityTracker)
    rl_tracker = RLMetricsTrackerV4(mock_tracker)

    mock_tracker.get_metrics.return_value = [
        {"value": 1.0, "skill_name": "search_skill"},
        {"value": 0.5, "skill_name": "other_skill"},
        {"value": 0.8, "skill_name": "search_skill"},
    ]

    trend = rl_tracker.get_reward_trend("search_skill", limit=2)

    assert len(trend) == 2
    assert trend[0]["value"] == 1.0
    assert trend[1]["value"] == 0.8
    mock_tracker.get_metrics.assert_called_with("rl_reward_v4", limit=10)

def test_calculate_average_reward() -> None:
    """Tests that average reward calculation is correct for a given skill."""
    mock_tracker = MagicMock(spec=QualityTracker)
    rl_tracker = RLMetricsTrackerV4(mock_tracker)

    mock_tracker.get_metrics.return_value = [
        {"value": 1.0, "skill_name": "search_skill"},
        {"value": 0.8, "skill_name": "search_skill"},
    ]

    avg = rl_tracker.calculate_average_reward("search_skill", limit=10)

    assert avg == 0.9

def test_calculate_average_reward_none() -> None:
    """Tests that average reward returns None when no metrics are found."""
    mock_tracker = MagicMock(spec=QualityTracker)
    rl_tracker = RLMetricsTrackerV4(mock_tracker)

    mock_tracker.get_metrics.return_value = []

    avg = rl_tracker.calculate_average_reward("non_existent_skill")

    assert avg is None
