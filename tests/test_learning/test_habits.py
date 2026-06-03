import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from magda_agent.learning.habits import HabitTracker
from magda_agent.planning.planner import Planner
from magda_agent.skills.registry import SkillRegistry
from magda_agent.llm_client import LLMClient

@pytest.fixture
def habit_tracker():
    return HabitTracker()

def test_habit_recording_and_suggestion(habit_tracker):
    # Initially no suggestion
    assert habit_tracker.suggest_strategy("test input") is None

    # Low score - should not form a habit
    habit_tracker.record_usage("test input", "skill_A", 6.0)
    assert habit_tracker.suggest_strategy("test input") is None

    # High score - one usage is not enough to form a strong habit (threshold is 2)
    habit_tracker.record_usage("test input", "skill_A", 9.0)
    assert habit_tracker.suggest_strategy("test input") is None

    # Second high score - should form a habit
    habit_tracker.record_usage("test input", "skill_A", 9.5)
    assert habit_tracker.suggest_strategy("test input") == "skill_A"

    # Another skill gets used successfully, but skill_A is still top
    habit_tracker.record_usage("test input", "skill_B", 8.5)
    assert habit_tracker.suggest_strategy("test input") == "skill_A"

@pytest.mark.asyncio
async def test_planner_integration(habit_tracker):
    # Setup mock LLM and SkillRegistry
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_completion = AsyncMock(return_value=json.dumps([{"description": "Do X", "skill": "skill_A"}]))

    mock_skills = MagicMock(spec=SkillRegistry)
    mock_skills.get_skills_summary.return_value = "skill_A: Does A"

    planner = Planner(llm=mock_llm, skills=mock_skills, habit_tracker=habit_tracker)

    # Setup a habit
    habit_tracker.record_usage("do A", "skill_A", 9.0)
    habit_tracker.record_usage("do A", "skill_A", 9.0)

    # Generate plan
    plan = await planner.generate_plan("do A")

    # Verify the LLM was called with the suggested strategy in the prompt
    mock_llm.chat_completion.assert_called_once()
    call_args = mock_llm.chat_completion.call_args[0][0]

    system_prompt = call_args[0]["content"]
    assert "Suggested strategy based on past success: consider using the 'skill_A' skill." in system_prompt
