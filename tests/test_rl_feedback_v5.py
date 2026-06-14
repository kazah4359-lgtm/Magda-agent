import pytest
from unittest.mock import MagicMock
from magda_agent.learning.rl_feedback import OnlineRLFeedbackLoop
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons

@pytest.fixture
def mock_habit_tracker():
    return MagicMock(spec=HabitTracker)

@pytest.fixture
def mock_mirror_neurons():
    return MagicMock(spec=MirrorNeurons)

@pytest.mark.asyncio
async def test_adjust_behavior_positive(mock_habit_tracker, mock_mirror_neurons):
    mock_mirror_neurons.empathize.return_value = (0.5, 0.1, 0.0) # score = (0.5+1)*5 = 7.5
    rl_loop = OnlineRLFeedbackLoop(mock_habit_tracker, mock_mirror_neurons)

    await rl_loop.adjust_behavior("Good job", "action ctx", 1, skill_used="test_skill")

    assert rl_loop.reward_parameters["test_skill"] == 1.2
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="action ctx", skill_used="test_skill", evaluation_score=7.5, user_id=1
    )

@pytest.mark.asyncio
async def test_adjust_behavior_negative(mock_habit_tracker, mock_mirror_neurons):
    mock_mirror_neurons.empathize.return_value = (-0.5, 0.1, 0.0) # score = (-0.5+1)*5 = 2.5
    rl_loop = OnlineRLFeedbackLoop(mock_habit_tracker, mock_mirror_neurons)

    await rl_loop.adjust_behavior("Bad job", "action ctx", 1, skill_used="test_skill")

    assert rl_loop.reward_parameters["test_skill"] == 0.8
    mock_habit_tracker.record_usage.assert_not_called()

@pytest.mark.asyncio
async def test_adjust_behavior_explicit_score(mock_habit_tracker, mock_mirror_neurons):
    mock_mirror_neurons.empathize.return_value = (0.0, 0.0, 0.0)
    rl_loop = OnlineRLFeedbackLoop(mock_habit_tracker, mock_mirror_neurons)
    await rl_loop.adjust_behavior("Okay", "action ctx", 1, explicit_score=8.0, skill_used="test_skill")

    assert rl_loop.reward_parameters["test_skill"] == 1.2
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="action ctx", skill_used="test_skill", evaluation_score=8.0, user_id=1
    )
