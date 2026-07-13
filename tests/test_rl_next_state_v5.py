import pytest
from unittest.mock import MagicMock
from magda_agent.learning.rl_next_state_v5 import RLNextStateSignalsV5
from magda_agent.learning.rl_metrics_v4 import RLMetricsTrackerV4
from magda_agent.metacognition.tracker import QualityTracker

@pytest.fixture
def mock_quality_tracker():
    tracker = MagicMock(spec=QualityTracker)
    return tracker

@pytest.fixture
def metrics_tracker(mock_quality_tracker):
    return RLMetricsTrackerV4(quality_tracker=mock_quality_tracker)

@pytest.fixture
def rl_module(metrics_tracker):
    return RLNextStateSignalsV5(metrics_tracker=metrics_tracker, alpha=0.1, gamma=0.9)

def test_parse_reward(rl_module):
    assert rl_module.parse_reward("This is great!") == 1.0
    assert rl_module.parse_reward("That is bad.") == -1.0
    assert rl_module.parse_reward("Just neutral.") == 0.0
    assert rl_module.parse_reward("Yes, it works.") == 1.0
    assert rl_module.parse_reward("No, it failed.") == -1.0

def test_process_signal_update(rl_module, mock_quality_tracker):
    skill_id = "test_skill"
    reward = 1.0
    next_state_max_q = 0.5

    # Q(s,a) = 0 + 0.1 * (1.0 + 0.9 * 0.5 - 0) = 0.1 * 1.45 = 0.145
    new_q = rl_module.process_signal(skill_id, reward, next_state_max_q)

    assert new_q == pytest.approx(0.145)
    assert rl_module.get_q_value(skill_id) == pytest.approx(0.145)

    # Verify logging
    assert mock_quality_tracker.log_metric.call_count == 2
    # First call for reward
    mock_quality_tracker.log_metric.assert_any_call("rl_reward_v4", 1.0, {"skill_name": skill_id, "user_id": None, "type": "reward"})
    # Second call for weight delta
    mock_quality_tracker.log_metric.assert_any_call("rl_weight_delta_v4", 0.145, {"skill_name": skill_id, "user_id": None, "type": "weight_delta"})

def test_handle_interaction(rl_module):
    skill_id = "chat_skill"
    # Q(s,a) = 0 + 0.1 * (1.0 + 0.9 * 0.0 - 0) = 0.1
    new_q = rl_module.handle_interaction(skill_id, "thanks a lot!")

    assert new_q == pytest.approx(0.1)
    assert rl_module.get_q_value(skill_id) == pytest.approx(0.1)

def test_multiple_updates(rl_module):
    skill_id = "multi_skill"

    # First update: Q = 0.1
    rl_module.handle_interaction(skill_id, "good")

    # Second update: reward = 1.0, next_q = 0, Q = 0.1 + 0.1 * (1.0 - 0.1) = 0.19
    new_q = rl_module.handle_interaction(skill_id, "great")

    assert new_q == pytest.approx(0.19)
