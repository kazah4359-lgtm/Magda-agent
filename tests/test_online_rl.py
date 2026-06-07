import pytest
from unittest.mock import MagicMock
from magda_agent.learning.online_rl import OnlineRLIntegrator
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
    mock_mirror_neurons.empathize.return_value = (0.8, 0.1, 0.0)
    integrator = OnlineRLIntegrator(mock_habit_tracker, mock_mirror_neurons)

    await integrator.process_feedback("Great job!", "test_context", user_id=1)

    # weight = (0.8 + 1.0) * 5.0 = 9.0
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="test_context",
        skill_used="rl_feedback_skill",
        evaluation_score=9.0,
        user_id=1
    )

@pytest.mark.asyncio
async def test_process_feedback_negative(mock_habit_tracker, mock_mirror_neurons):
    mock_mirror_neurons.empathize.return_value = (-0.5, 0.1, 0.0)
    integrator = OnlineRLIntegrator(mock_habit_tracker, mock_mirror_neurons)

    await integrator.process_feedback("Terrible.", "test_context", user_id=1)

    # weight = (-0.5 + 1.0) * 5.0 = 2.5 (less than 8.0)
    mock_habit_tracker.record_usage.assert_not_called()
