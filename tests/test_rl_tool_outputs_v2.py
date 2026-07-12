import pytest
from unittest.mock import AsyncMock, MagicMock

from magda_agent.learning.rl_tool_outputs_v2 import OpenClawRLToolOutputsV2
from magda_agent.learning.habits import HabitTracker
from magda_agent.user_model.model import UserModel

@pytest.fixture
def mock_habit_tracker():
    return MagicMock(spec=HabitTracker)

@pytest.fixture
def mock_user_model():
    model = MagicMock(spec=UserModel)
    model.get_model.return_value = {
        "preferences": {},
        "communication_style": "default",
        "expertise_level": "unknown",
        "recurring_topics": [],
        "skill_weights": {"weather_skill": 1.0}
    }
    return model

@pytest.mark.asyncio
async def test_successful_tool_output(mock_habit_tracker, mock_user_model):
    learner = OpenClawRLToolOutputsV2(
        habit_tracker=mock_habit_tracker,
        user_model=mock_user_model
    )

    await learner.process_tool_output_signal(
        tool_name="weather_skill",
        tool_output="The weather in London is 20C and sunny.",
        action_context="Get weather for London",
        user_id=123
    )

    # Check that usage was recorded (reward 8.0)
    mock_habit_tracker.record_usage.assert_called_once_with(
        input_text="Get weather for London",
        skill_used="weather_skill",
        evaluation_score=8.0,
        user_id=123
    )

    # Check that weight was increased from 1.0 to 1.2
    mock_user_model.save_model.assert_called_once()
    saved_model = mock_user_model.save_model.call_args[0][1]
    assert saved_model["skill_weights"]["weather_skill"] == 1.2
    assert saved_model["rl_tool_preferences"]["last_tool_reward"] == 8.0

@pytest.mark.asyncio
async def test_failed_tool_output(mock_habit_tracker, mock_user_model):
    learner = OpenClawRLToolOutputsV2(
        habit_tracker=mock_habit_tracker,
        user_model=mock_user_model
    )

    await learner.process_tool_output_signal(
        tool_name="weather_skill",
        tool_output="Error: API key invalid.",
        action_context="Get weather for London",
        user_id=123
    )

    # Check that usage was NOT recorded because reward is 2.0 (<= 5.0)
    mock_habit_tracker.record_usage.assert_not_called()

    # Check that weight was decreased from 1.0 to 0.8
    mock_user_model.save_model.assert_called_once()
    saved_model = mock_user_model.save_model.call_args[0][1]
    assert saved_model["skill_weights"]["weather_skill"] == 0.8
    assert saved_model["rl_tool_preferences"]["last_tool_reward"] == 2.0

@pytest.mark.asyncio
async def test_max_weight_cap(mock_habit_tracker, mock_user_model):
    # Set current weight to 1.9, it should cap at 2.0
    mock_user_model.get_model.return_value = {
        "skill_weights": {"weather_skill": 1.9}
    }

    learner = OpenClawRLToolOutputsV2(
        habit_tracker=mock_habit_tracker,
        user_model=mock_user_model
    )

    await learner.process_tool_output_signal(
        tool_name="weather_skill",
        tool_output="The weather in London is 20C and sunny.",
        action_context="Get weather for London",
        user_id=123
    )

    saved_model = mock_user_model.save_model.call_args[0][1]
    assert saved_model["skill_weights"]["weather_skill"] == 2.0

@pytest.mark.asyncio
async def test_min_weight_floor(mock_habit_tracker, mock_user_model):
    # Set current weight to 0.2, it should floor at 0.1
    mock_user_model.get_model.return_value = {
        "skill_weights": {"weather_skill": 0.2}
    }

    learner = OpenClawRLToolOutputsV2(
        habit_tracker=mock_habit_tracker,
        user_model=mock_user_model
    )

    await learner.process_tool_output_signal(
        tool_name="weather_skill",
        tool_output="Error: Timeout.",
        action_context="Get weather for London",
        user_id=123
    )

    saved_model = mock_user_model.save_model.call_args[0][1]
    # 0.2 - 0.2 = 0.0, but should floor at 0.1
    assert saved_model["skill_weights"]["weather_skill"] == 0.1

@pytest.mark.asyncio
async def test_empty_tool_output_returns_early(mock_habit_tracker, mock_user_model):
    learner = OpenClawRLToolOutputsV2(
        habit_tracker=mock_habit_tracker,
        user_model=mock_user_model
    )

    await learner.process_tool_output_signal(
        tool_name="weather_skill",
        tool_output="",
        action_context="Get weather for London",
        user_id=123
    )

    mock_habit_tracker.record_usage.assert_not_called()
    mock_user_model.save_model.assert_not_called()
