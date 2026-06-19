import pytest
from unittest.mock import MagicMock
from magda_agent.learning.rl_signals import RLSignalProcessor
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons

@pytest.fixture
def mock_habit_tracker():
    return MagicMock(spec=HabitTracker)

@pytest.fixture
def mock_mirror_neurons():
    return MagicMock(spec=MirrorNeurons)

@pytest.mark.asyncio
async def test_rl_signals_positive_feedback(mock_habit_tracker, mock_mirror_neurons):
    mock_mirror_neurons.empathize.return_value = (0.5, 0.0, 0.0)
    processor = RLSignalProcessor(mock_habit_tracker, mock_mirror_neurons)

    await processor.process_signal("Great job", "context", user_id=1, skills_used=["test_skill"])

    mock_mirror_neurons.empathize.assert_called_once_with("Great job")

    # Expected score calculation:
    # p_shift = 0.5
    # score = (0.5 + 1.0) * 5.0 = 7.5
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="context",
        skill_used="test_skill",
        evaluation_score=7.5,
        user_id=1
    )

@pytest.mark.asyncio
async def test_rl_signals_negative_feedback(mock_habit_tracker, mock_mirror_neurons):
    mock_mirror_neurons.empathize.return_value = (-0.5, 0.0, 0.0)
    processor = RLSignalProcessor(mock_habit_tracker, mock_mirror_neurons)

    await processor.process_signal("This is wrong", "context", user_id=1)

    mock_mirror_neurons.empathize.assert_called_once_with("This is wrong")
    mock_habit_tracker.record_usage.assert_not_called()

@pytest.mark.asyncio
async def test_rl_signals_with_tool_output(mock_habit_tracker, mock_mirror_neurons):
    mock_mirror_neurons.empathize.return_value = (0.8, 0.0, 0.0)
    processor = RLSignalProcessor(mock_habit_tracker, mock_mirror_neurons)

    await processor.process_signal("Nice", "context", user_id=2, tool_output="Tool success")

    mock_mirror_neurons.empathize.assert_called_once_with("Nice [Tool Output: Tool success]")

    # Expected score calculation:
    # p_shift = 0.8
    # score = (0.8 + 1.0) * 5.0 + 1.0 = 10.0
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="context",
        skill_used="rl_skill",
        evaluation_score=10.0,
        user_id=2
    )
