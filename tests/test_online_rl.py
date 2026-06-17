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

    await integrator.process_feedback("Great job!", "test_context", user_id=1, skill_used="rl_feedback_skill")

    # weight = (0.8 + 1.0) * 5.0 = 9.0
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="test_context",
        skill_used="rl_feedback_skill",
        evaluation_score=9.0,
        user_id=1
    )
    assert integrator.skill_weights["rl_feedback_skill"] == 1.1

@pytest.mark.asyncio
async def test_process_feedback_negative(mock_habit_tracker, mock_mirror_neurons):
    mock_mirror_neurons.empathize.return_value = (-0.5, 0.1, 0.0)
    integrator = OnlineRLIntegrator(mock_habit_tracker, mock_mirror_neurons)

    await integrator.process_feedback("Terrible.", "test_context", user_id=1, skill_used="rl_feedback_skill")

    # weight = (-0.5 + 1.0) * 5.0 = 2.5 (less than 8.0)
    mock_habit_tracker.record_usage.assert_not_called()
    assert integrator.skill_weights["rl_feedback_skill"] == 0.9


from magda_agent.learning.online_rl import OnlineRLLearner

@pytest.mark.asyncio
async def test_online_rl_learner(mock_habit_tracker, mock_mirror_neurons):
    mock_mirror_neurons.empathize.return_value = (0.5, 0.1, 0.0)
    learner = OnlineRLLearner(mock_habit_tracker, mock_mirror_neurons)

    await learner.learn_from_feedback("This is fine", "test_context", user_id=3, tool_output="Success: data saved", tool_success=True, skill_used="rl_skill")

    # The integrator parses "This is fine [Tool Output: Success: data saved]"
    mock_mirror_neurons.empathize.assert_called_once_with("This is fine [Tool Output: Success: data saved]")

    # weight = (0.5 + 1.0) * 5.0 + 2.0 (tool_success) = 7.5 + 2.0 = 9.5
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="test_context",
        skill_used="rl_skill",
        evaluation_score=9.5,
        user_id=3
    )
