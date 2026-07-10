import pytest
from unittest.mock import MagicMock
from magda_agent.learning.rl_user_behavior_v4 import OnlineRLUserBehaviorV4
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons

@pytest.mark.asyncio
async def test_process_session_metrics_positive():
    """
    Test that high frequency and long duration result in positive reinforcement.
    """
    habit_tracker_mock = MagicMock(spec=HabitTracker)
    mirror_neurons_mock = MagicMock(spec=MirrorNeurons)

    learner = OnlineRLUserBehaviorV4(habit_tracker=habit_tracker_mock, mirror_neurons=mirror_neurons_mock)

    # 20 interactions over 15 minutes (900 seconds) -> rate = 1.33
    # reward = 1.33 * 2.0 = 2.66 + 2.0 (bonus for > 10 min) = 4.66 -> still low? Let's increase rate.
    # 50 interactions over 15 minutes (900 seconds) -> rate = 3.33
    # reward = 3.33 * 2.0 = 6.66 + 2.0 (bonus) = 8.66
    await learner.process_session_metrics(
        frequency=50,
        duration_seconds=900.0,
        skill_name="test_skill",
        user_id=1
    )

    habit_tracker_mock.record_usage.assert_called_once()
    args, kwargs = habit_tracker_mock.record_usage.call_args
    assert kwargs["skill_used"] == "test_skill"
    assert kwargs["user_id"] == 1
    assert kwargs["evaluation_score"] > 5.0

@pytest.mark.asyncio
async def test_process_session_metrics_low_engagement():
    """
    Test that low engagement (short duration, few interactions) does not reinforce.
    """
    habit_tracker_mock = MagicMock(spec=HabitTracker)
    mirror_neurons_mock = MagicMock(spec=MirrorNeurons)

    learner = OnlineRLUserBehaviorV4(habit_tracker=habit_tracker_mock, mirror_neurons=mirror_neurons_mock)

    # 2 interactions over 5 minutes (300 seconds) -> rate = 0.4
    # reward = 0.4 * 2.0 = 0.8
    await learner.process_session_metrics(
        frequency=2,
        duration_seconds=300.0,
        skill_name="test_skill",
        user_id=1
    )

    habit_tracker_mock.record_usage.assert_not_called()

@pytest.mark.asyncio
async def test_process_session_metrics_invalid_inputs():
    """
    Test that zero or negative inputs are handled gracefully.
    """
    habit_tracker_mock = MagicMock(spec=HabitTracker)
    mirror_neurons_mock = MagicMock(spec=MirrorNeurons)

    learner = OnlineRLUserBehaviorV4(habit_tracker=habit_tracker_mock, mirror_neurons=mirror_neurons_mock)

    await learner.process_session_metrics(
        frequency=0,
        duration_seconds=100.0,
        skill_name="test_skill",
        user_id=1
    )

    habit_tracker_mock.record_usage.assert_not_called()

    await learner.process_session_metrics(
        frequency=10,
        duration_seconds=0.0,
        skill_name="test_skill",
        user_id=1
    )

    habit_tracker_mock.record_usage.assert_not_called()
