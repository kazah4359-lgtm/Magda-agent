import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.learning.online import OnlineLearner
from magda_agent.emotions.engine import PADState

@pytest.mark.asyncio
async def test_online_learner_positive_feedback():
    habit_tracker = MagicMock()
    memory = MagicMock()
    mirror_neurons = MagicMock()
    # Empathize returns positive shift
    mirror_neurons.empathize.return_value = (0.2, 0.1, 0.0)

    learner = OnlineLearner(habit_tracker, memory, mirror_neurons)
    await learner.process_feedback("Thanks, that was great!", "Some context", user_id=42)

    # Should reinforce habit
    habit_tracker.record_usage.assert_called_once_with("Some context", "online_learned_skill", 10.0, user_id=42)
    # Should not add a negative reflection memory
    memory.add_memory.assert_not_called()

@pytest.mark.asyncio
async def test_online_learner_negative_feedback():
    habit_tracker = MagicMock()
    memory = MagicMock()
    memory.add_memory = AsyncMock()
    mirror_neurons = MagicMock()
    # Empathize returns negative shift
    mirror_neurons.empathize.return_value = (-0.3, 0.2, 0.1)

    learner = OnlineLearner(habit_tracker, memory, mirror_neurons)
    await learner.process_feedback("This is completely wrong and useless.", "Some bad context", user_id=42)

    # Should not reinforce habit
    habit_tracker.record_usage.assert_not_called()
    # Should add a reflection memory
    memory.add_memory.assert_called_once()

    call_kwargs = memory.add_memory.call_args.kwargs
    assert "Negative feedback received" in call_kwargs["content"]
    assert "Some bad context" in call_kwargs["content"]
    assert "reflection" in call_kwargs["tags"]
    assert call_kwargs["user_id"] == 42
