import pytest
import os
import json
import tempfile
from unittest.mock import MagicMock
from magda_agent.learning.openclaw_rl import OpenClawInteractiveLearner
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons
from magda_agent.user_model.model import UserModel
from magda_agent.learning.lessons import TaskRecoveryLessons

@pytest.fixture
def mock_habit_tracker():
    tracker = MagicMock(spec=HabitTracker)
    return tracker

@pytest.fixture
def mock_mirror_neurons():
    neurons = MagicMock(spec=MirrorNeurons)
    return neurons

@pytest.fixture
def mock_recovery_lessons():
    lessons = MagicMock(spec=TaskRecoveryLessons)
    return lessons

@pytest.fixture
def temp_user_model_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def user_model(temp_user_model_dir):
    model = UserModel(persist_dir=temp_user_model_dir, llm=None)
    return model

@pytest.mark.asyncio
async def test_openclaw_rl_positive_signal(mock_habit_tracker: MagicMock, mock_mirror_neurons: MagicMock, user_model: UserModel) -> None:
    """Tests that a positive next-state signal reinforces habits and updates preferences."""
    learner = OpenClawInteractiveLearner(mock_habit_tracker, mock_mirror_neurons, user_model)

    # Mock positive empathize
    mock_mirror_neurons.empathize.return_value = (0.2, 0.1, 0.0)

    await learner.process_next_state_signal("Great job", "test_context", 1, skills_used=["skill_a", "skill_b"])

    # Check habit tracker called for both skills
    assert mock_habit_tracker.record_usage.call_count == 2
    mock_habit_tracker.record_usage.assert_any_call(
        input_text="test_context", skill_used="skill_a", evaluation_score=10.0, user_id=1
    )
    mock_habit_tracker.record_usage.assert_any_call(
        input_text="test_context", skill_used="skill_b", evaluation_score=10.0, user_id=1
    )

    # Check user model modification
    model_data = user_model.get_model(1)
    assert model_data["preferences"]["last_p_shift"] == 0.2
    assert "(friendly)" in model_data["communication_style"]

@pytest.mark.asyncio
async def test_openclaw_rl_negative_signal(mock_habit_tracker: MagicMock, mock_mirror_neurons: MagicMock, user_model: UserModel, mock_recovery_lessons: MagicMock) -> None:
    """Tests that a negative next-state signal generates recovery lessons and updates preferences."""
    learner = OpenClawInteractiveLearner(
        mock_habit_tracker, mock_mirror_neurons, user_model, recovery_lessons=mock_recovery_lessons
    )

    # Mock negative empathize
    mock_mirror_neurons.empathize.return_value = (-0.3, 0.1, 0.1)

    await learner.process_next_state_signal("This is terrible", "test_context", 2)

    # Check habit tracker NOT called
    mock_habit_tracker.record_usage.assert_not_called()

    # Check recovery lesson generation
    mock_recovery_lessons.generate_and_store_lesson.assert_called_once_with(
        task_description="test_context",
        failure_reason="This is terrible",
        user_id=2
    )

    # Check user model modification
    model_data = user_model.get_model(2)
    assert model_data["preferences"]["last_p_shift"] == -0.3
    assert "(cautious)" in model_data["communication_style"]

@pytest.mark.asyncio
async def test_openclaw_rl_tool_output(mock_habit_tracker: MagicMock, mock_mirror_neurons: MagicMock, user_model: UserModel) -> None:
    """Tests that the OpenClawInteractiveLearner correctly processes next-state signals with tool outputs."""
    learner = OpenClawInteractiveLearner(mock_habit_tracker, mock_mirror_neurons, user_model)

    # Mock positive empathize
    mock_mirror_neurons.empathize.return_value = (0.5, 0.1, 0.0)

    await learner.process_next_state_signal("This is fine", "test_context", 3, tool_output="Success: data saved")

    # Check mirror neurons called with both
    mock_mirror_neurons.empathize.assert_called_once_with("This is fine [Tool Output: Success: data saved]")

    # Check habit tracker
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="test_context", skill_used="rl_skill", evaluation_score=10.0, user_id=3
    )

    # Check user model modification
    model_data = user_model.get_model(3)
    assert model_data["preferences"]["last_p_shift"] == 0.5
    assert "(friendly)" in model_data["communication_style"]
