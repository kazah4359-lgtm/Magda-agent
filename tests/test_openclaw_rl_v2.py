import pytest
from unittest.mock import MagicMock

from magda_agent.learning.openclaw_rl_v2 import OpenClawRLOnlineLearnerV2

@pytest.fixture
def mock_habit_tracker() -> MagicMock:
    """Mocks the HabitTracker."""
    return MagicMock()

@pytest.fixture
def mock_mirror_neurons() -> MagicMock:
    """Mocks the MirrorNeurons."""
    return MagicMock()

@pytest.fixture
def mock_user_model() -> MagicMock:
    """Mocks the UserModel."""
    return MagicMock()

@pytest.fixture
def learner(mock_habit_tracker: MagicMock, mock_mirror_neurons: MagicMock, mock_user_model: MagicMock) -> OpenClawRLOnlineLearnerV2:
    """Provides a fresh OpenClawRLOnlineLearnerV2 instance."""
    return OpenClawRLOnlineLearnerV2(
        habit_tracker=mock_habit_tracker,
        mirror_neurons=mock_mirror_neurons,
        user_model=mock_user_model
    )

@pytest.mark.asyncio
async def test_process_positive_signal(

    learner: OpenClawRLOnlineLearnerV2,
    mock_habit_tracker: MagicMock,
    mock_mirror_neurons: MagicMock,
    mock_user_model: MagicMock
) -> None:
    """Tests positive signal."""
    # Arrange
    user_id = 42
    user_reply = "That was very helpful!"
    action_context = "Completed task XYZ"
    tool_output = "Task XYZ completed successfully"

    mock_mirror_neurons.empathize.return_value = (0.5, 0.1, 0.0) # p_shift = 0.5
    mock_user_model.get_model.return_value = {"communication_style": "default"}

    # Act
    await learner.process_next_state_signal(
        user_reply=user_reply,
        action_context=action_context,
        user_id=user_id,
        tool_output=tool_output
    )

    # Assert
    # Base reward = (0.5 + 1.0) * 5.0 = 7.5
    # Bonus = +2.0 (since tool_output is present and p_shift > 0)
    # Total reward = 9.5
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text=action_context,
        skill_used="rl_skill_v2",
        evaluation_score=9.5,
        user_id=user_id
    )

    saved_model = mock_user_model.save_model.call_args[0][1]
    assert "(confident)" in saved_model["communication_style"]
    assert saved_model["rl_v2_preferences"]["last_reward"] == 9.5

@pytest.mark.asyncio
async def test_process_negative_signal(

    learner: OpenClawRLOnlineLearnerV2,
    mock_habit_tracker: MagicMock,
    mock_mirror_neurons: MagicMock,
    mock_user_model: MagicMock
) -> None:
    """Tests positive signal."""
    # Arrange
    user_id = 42
    user_reply = "This didn't work at all"
    action_context = "Failed attempt at XYZ"

    mock_mirror_neurons.empathize.return_value = (-0.8, -0.2, 0.0) # p_shift = -0.8
    mock_user_model.get_model.return_value = {"communication_style": "default"}

    # Act
    await learner.process_next_state_signal(
        user_reply=user_reply,
        action_context=action_context,
        user_id=user_id,
        tool_output=None
    )

    # Assert
    # Base reward = (-0.8 + 1.0) * 5.0 = 1.0
    # No bonus since tool output is None
    # Total reward = 1.0
    mock_habit_tracker.record_usage.assert_not_called()

    saved_model = mock_user_model.save_model.call_args[0][1]
    assert "(attentive)" in saved_model["communication_style"]
    assert abs(saved_model["rl_v2_preferences"]["last_reward"] - 1.0) < 1e-5

@pytest.mark.asyncio
async def test_empty_signals(

    learner: OpenClawRLOnlineLearnerV2,
    mock_habit_tracker: MagicMock,
    mock_mirror_neurons: MagicMock,
    mock_user_model: MagicMock
) -> None:
    """Tests empty signals do not process anything."""
    # Act
    await learner.process_next_state_signal(
        user_reply="",
        action_context="Something",
        user_id=1
    )

    # Assert
    mock_mirror_neurons.empathize.assert_not_called()
    mock_user_model.get_model.assert_not_called()
    mock_habit_tracker.record_usage.assert_not_called()
    mock_user_model.save_model.assert_not_called()
