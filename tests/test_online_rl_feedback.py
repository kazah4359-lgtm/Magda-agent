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
async def test_process_feedback_implicit_positive(mock_habit_tracker, mock_mirror_neurons):
    mock_mirror_neurons.empathize.return_value = (0.8, 0.1, 0.0)
    integrator = OnlineRLIntegrator(mock_habit_tracker, mock_mirror_neurons)

    await integrator.process_feedback("Great job!", "test_context", user_id=1)

    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="test_context",
        skill_used="rl_feedback_skill",
        evaluation_score=9.0,
        user_id=1
    )

@pytest.mark.asyncio
async def test_process_feedback_explicit_score(mock_habit_tracker, mock_mirror_neurons):
    mock_mirror_neurons.empathize.return_value = (0.0, 0.0, 0.0)
    integrator = OnlineRLIntegrator(mock_habit_tracker, mock_mirror_neurons)

    await integrator.process_feedback("Okay", "test_context", explicit_score=9.0, skill_used="custom_skill")

    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="test_context",
        skill_used="custom_skill",
        evaluation_score=9.0,
        user_id=None
    )

@pytest.mark.asyncio
async def test_process_feedback_tool_success(mock_habit_tracker, mock_mirror_neurons):
    mock_mirror_neurons.empathize.return_value = (0.5, 0.0, 0.0)
    integrator = OnlineRLIntegrator(mock_habit_tracker, mock_mirror_neurons)
    # base_score = 7.5. tool_success + 2.0 = 9.5
    await integrator.process_feedback("Good", "test_context", tool_success=True)

    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="test_context",
        skill_used="rl_feedback_skill",
        evaluation_score=9.5,
        user_id=None
    )
