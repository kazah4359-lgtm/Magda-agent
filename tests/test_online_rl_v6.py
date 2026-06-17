import pytest
from unittest.mock import MagicMock
from magda_agent.learning.online_rl_v6 import OnlineRLFeedbackLoopV6
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons

@pytest.fixture
def mock_habit_tracker():
    tracker = MagicMock(spec=HabitTracker)
    return tracker

@pytest.fixture
def mock_mirror_neurons():
    neurons = MagicMock(spec=MirrorNeurons)
    return neurons

@pytest.mark.asyncio
async def test_online_rl_v6_positive_adjustment(mock_habit_tracker, mock_mirror_neurons):
    learner = OnlineRLFeedbackLoopV6(mock_habit_tracker, mock_mirror_neurons)

    # Mock positive empathize resulting in high score. In reality, empathize returns Tuple[float, float, float]
    # For a score of 9.0 without explicit score, (p_shift + 1.0) * 5.0 = 9.0 => p_shift = 0.8
    mock_mirror_neurons.empathize.return_value = (0.8, 0.1, 0.0)

    await learner.adjust_behavior("Great job", "test_context", user_id=1)

    # Check habit tracker called
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="test_context", skill_used="rl_feedback_skill", evaluation_score=9.0, user_id=1
    )

    # Check reward parameter
    assert learner.reward_parameters["rl_feedback_skill"] == 1.2

@pytest.mark.asyncio
async def test_online_rl_v6_negative_adjustment(mock_habit_tracker, mock_mirror_neurons):
    learner = OnlineRLFeedbackLoopV6(mock_habit_tracker, mock_mirror_neurons)

    # Mock negative empathize resulting in low score. For score < 7.0, say p_shift = -0.5
    # score = (-0.5 + 1.0) * 5.0 = 2.5
    mock_mirror_neurons.empathize.return_value = (-0.5, 0.1, 0.1)

    await learner.adjust_behavior("Terrible", "test_context", user_id=2)

    # Check habit tracker NOT called
    mock_habit_tracker.record_usage.assert_not_called()

    # Check reward parameter decreased
    assert learner.reward_parameters["rl_feedback_skill"] == 0.8
