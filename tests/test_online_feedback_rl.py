import pytest
from unittest.mock import MagicMock, AsyncMock
from magda_agent.learning.online_feedback_rl import OnlineFeedbackRL
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons

@pytest.fixture
def mock_habit_tracker():
    return MagicMock(spec=HabitTracker)

@pytest.fixture
def mock_mirror_neurons():
    return MagicMock(spec=MirrorNeurons)

@pytest.mark.asyncio
async def test_process_feedback_positive(mock_habit_tracker, mock_mirror_neurons):
    # P shift 0.8 -> Reward (0.8 + 1.0) * 5 = 9.0
    mock_mirror_neurons.empathize.return_value = (0.8, 0.0, 0.0)
    rl = OnlineFeedbackRL(mock_habit_tracker, mock_mirror_neurons)

    await rl.process_feedback("Awesome!", "context", ["test_skill"], user_id=123)

    assert rl.get_skill_weight("test_skill") == 1.1
    mock_habit_tracker.record_usage.assert_called_once()

@pytest.mark.asyncio
async def test_process_feedback_negative(mock_habit_tracker, mock_mirror_neurons):
    # P shift -0.6 -> Reward (-0.6 + 1.0) * 5 = 2.0
    mock_mirror_neurons.empathize.return_value = (-0.6, 0.0, 0.0)
    rl = OnlineFeedbackRL(mock_habit_tracker, mock_mirror_neurons)

    await rl.process_feedback("Bad.", "context", ["test_skill"])

    assert rl.get_skill_weight("test_skill") == 0.8
    mock_habit_tracker.record_usage.assert_not_called()

@pytest.mark.asyncio
async def test_multiple_skills(mock_habit_tracker, mock_mirror_neurons):
    mock_mirror_neurons.empathize.return_value = (0.5, 0.0, 0.0) # Reward 7.5
    rl = OnlineFeedbackRL(mock_habit_tracker, mock_mirror_neurons)

    await rl.process_feedback("OK", "context", ["skill1", "skill2"])

    assert rl.get_skill_weight("skill1") == 1.1
    assert rl.get_skill_weight("skill2") == 1.1
    # Habit tracker only called for >= 8.5
    mock_habit_tracker.record_usage.assert_not_called()
