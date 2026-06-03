import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from magda_agent.planning.planner import Planner
from magda_agent.llm_client import LLMClient
from magda_agent.skills.registry import SkillRegistry

@pytest.mark.asyncio
async def test_generate_plan_success():
    mock_llm = MagicMock(spec=LLMClient)

    mock_plan = [
        {"description": "Step 1", "skill": "search_internet", "skill_kwargs": {"query": "test query"}},
        {"description": "Step 2", "skill": None, "skill_kwargs": None}
    ]

    mock_llm.chat_completion = AsyncMock(return_value=json.dumps(mock_plan))

    mock_skills = MagicMock(spec=SkillRegistry)
    mock_skills.get_skills_summary.return_value = "Available Skills..."

    planner = Planner(llm=mock_llm, skills=mock_skills)

    result = await planner.generate_plan("Do something complex")

    assert len(result) == 2
    assert result[0]["description"] == "Step 1"
    assert result[1]["skill"] is None

    assert len(planner.current_plan) == 2
    assert len(planner.completed_steps) == 0

@pytest.mark.asyncio
async def test_generate_plan_invalid_json():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.chat_completion = AsyncMock(return_value="I am an AI, I can't generate JSON")

    mock_skills = MagicMock(spec=SkillRegistry)

    planner = Planner(llm=mock_llm, skills=mock_skills)

    result = await planner.generate_plan("Invalid request")

    assert result == []
    assert len(planner.current_plan) == 0

def test_mark_step_completed():
    mock_llm = MagicMock(spec=LLMClient)
    mock_skills = MagicMock(spec=SkillRegistry)
    planner = Planner(llm=mock_llm, skills=mock_skills)

    planner.current_plan = [
        {"description": "Step 1", "skill": "search_internet"},
        {"description": "Step 2", "skill": None}
    ]

    planner.mark_step_completed(0, "Search successful")

    assert len(planner.current_plan) == 1
    assert planner.current_plan[0]["description"] == "Step 2"

    assert len(planner.completed_steps) == 1
    assert planner.completed_steps[0]["description"] == "Step 1"
    assert planner.completed_steps[0]["result"] == "Search successful"

@pytest.mark.asyncio
async def test_consciousness_executes_plan():
    from magda_agent.consciousness.core import Consciousness
    from magda_agent.emotions.engine import EmotionalEngine
    from magda_agent.memory.storage import MemorySystem

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.get_system_prompt.return_value = "System prompt"
    mock_llm.chat_completion = AsyncMock(return_value="Final LLM Response")

    mock_skills = MagicMock(spec=SkillRegistry)
    # the skill will return a mocked string
    mock_skills.execute_skill.return_value = "Mocked search result for testing"
    mock_skills.get_skills_summary.return_value = "Available Skills..."

    mock_planner = MagicMock(spec=Planner)

    # current_plan initially has 2 steps, then 1, then 0. We'll simulate this with a side effect or just use a real Planner instance to be safer.
    real_planner = Planner(llm=mock_llm, skills=mock_skills)
    real_planner.generate_plan = AsyncMock()

    # set up the initial plan
    real_planner.current_plan = [
        {"description": "Step 1", "skill": "search_internet", "skill_kwargs": {"query": "test query"}},
        {"description": "Step 2", "skill": None, "skill_kwargs": None}
    ]

    emotions = EmotionalEngine()
    memory = MemorySystem()

    consciousness = Consciousness(
        llm=mock_llm,
        emotions=emotions,
        memory=memory,
        skills=mock_skills,
        planner=real_planner
    )

    await consciousness.process_input("Help me test")

    # verify the skill was executed
    mock_skills.execute_skill.assert_called_once_with("search_internet", query="test query")

    # verify completed steps have the result
    assert len(real_planner.completed_steps) == 2
    assert real_planner.completed_steps[0]["result"] == "Mocked search result for testing"
    assert real_planner.completed_steps[1]["result"] == "No skill executed for this step."

def test_get_state_summary():
    mock_llm = MagicMock(spec=LLMClient)
    mock_skills = MagicMock(spec=SkillRegistry)
    planner = Planner(llm=mock_llm, skills=mock_skills)

    planner.current_plan = [{"description": "Pending step", "skill": None}]
    planner.completed_steps = [{"description": "Done step", "skill": "search", "result": "Found it"}]

    summary = planner.get_state_summary()

    assert "Pending step" in summary
    assert "Done step" in summary
    assert "Found it" in summary
