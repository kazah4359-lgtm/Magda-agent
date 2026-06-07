import pytest
from unittest.mock import MagicMock
from magda_agent.learning.feedback_loop import FeedbackLoop

@pytest.mark.asyncio
async def test_feedback_loop_positive_feedback():
    mock_mirror_neurons = MagicMock()
    mock_mirror_neurons.empathize.return_value = (0.2, 0.1, 0.0) # Positive p_shift

    mock_user_model = MagicMock()
    mock_user_model.get_model.return_value = {"communication_style": "default"}

    feedback_loop = FeedbackLoop(mirror_neurons=mock_mirror_neurons, user_model=mock_user_model)
    await feedback_loop.process_feedback("Great job!", 123)

    call_args = mock_user_model.save_model.call_args[0]
    user_id = call_args[0]
    model_data = call_args[1]

    assert user_id == 123
    assert "(friendly)" in model_data["communication_style"]
    assert model_data["preferences"]["last_p_shift"] == 0.2

@pytest.mark.asyncio
async def test_feedback_loop_negative_feedback():
    mock_mirror_neurons = MagicMock()
    mock_mirror_neurons.empathize.return_value = (-0.3, 0.1, 0.0) # Negative p_shift

    mock_user_model = MagicMock()
    mock_user_model.get_model.return_value = {"communication_style": "default"}

    feedback_loop = FeedbackLoop(mirror_neurons=mock_mirror_neurons, user_model=mock_user_model)
    await feedback_loop.process_feedback("This is awful", 123)

    call_args = mock_user_model.save_model.call_args[0]
    user_id = call_args[0]
    model_data = call_args[1]

    assert user_id == 123
    assert "(cautious)" in model_data["communication_style"]
    assert model_data["preferences"]["last_p_shift"] == -0.3

@pytest.mark.asyncio
async def test_feedback_loop_no_feedback():
    mock_mirror_neurons = MagicMock()
    mock_user_model = MagicMock()

    feedback_loop = FeedbackLoop(mirror_neurons=mock_mirror_neurons, user_model=mock_user_model)
    await feedback_loop.process_feedback("", 123)

    mock_mirror_neurons.empathize.assert_not_called()
    mock_user_model.save_model.assert_not_called()
