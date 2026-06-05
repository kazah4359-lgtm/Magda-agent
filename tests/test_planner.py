import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from magda_agent.planning.planner import Planner
from magda_agent.llm_client import LLMClient
from magda_agent.skills.registry import SkillRegistry

@pytest.mark.asyncio
async def test_generate_plan_success():
    mock_llm = MagicMock(spec=LLMClient)

    mock_plan = {
        "goal": "Do something complex",
        "constraints": ["no new dependencies"],
        "risk": "low",
        "steps": [
            {"description": "Step 1", "skill": "search_internet", "skill_kwargs": {"query": "test query"}},
            {"description": "Step 2", "skill": None, "skill_kwargs": None}
        ],
        "acceptance": ["test passes"]
    }

    mock_llm.chat_completion = AsyncMock(return_value=json.dumps(mock_plan))

    mock_skills = MagicMock(spec=SkillRegistry)
    mock_skills.get_skills_summary.return_value = "Available Skills..."

    planner = Planner(llm=mock_llm, skills=mock_skills)

    result = await planner.generate_plan("Do something complex")

    assert len(result) == 2
    assert result[0]["description"] == "Step 1"
    assert result[1]["skill"] is None

    assert planner.current_goal == "Do something complex"
    assert planner.current_risk == "low"
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

@pytest.mark.asyncio
async def test_generate_plan_validation_failures():
    """
    Tests that generate_plan correctly validates the LLM-generated plan output.
    It verifies that a plan is rejected and an empty list is returned if the plan
    does not match the TypedPlan schema, contains invalid items, is missing keys,
    references an unknown skill, or contains invalid skill arguments.
    """
    mock_llm = MagicMock(spec=LLMClient)
    mock_skills = MagicMock(spec=SkillRegistry)
    mock_skills.has_skill.return_value = False

    planner = Planner(llm=mock_llm, skills=mock_skills)

    # 1. Not a JSON object matching TypedPlan
    mock_llm.chat_completion = AsyncMock(return_value='{"step": 1}')
    result = await planner.generate_plan("test")
    assert result == []

    # 2. Step is not a valid PlanStep dict
    mock_plan = {
        "goal": "Test", "risk": "low", "steps": ["step 1"]
    }
    mock_llm.chat_completion = AsyncMock(return_value=json.dumps(mock_plan))
    result = await planner.generate_plan("test")
    assert result == []

    # 3. Missing required keys in TypedPlan (e.g., missing 'goal')
    mock_plan = {
        "risk": "low", "steps": [{"description": "desc"}]
    }
    mock_llm.chat_completion = AsyncMock(return_value=json.dumps(mock_plan))
    result = await planner.generate_plan("test")
    assert result == []

    # 4. Missing required keys in PlanStep
    mock_plan = {
        "goal": "Test", "risk": "low", "steps": [{"skill": "known"}]
    }
    mock_llm.chat_completion = AsyncMock(return_value=json.dumps(mock_plan))
    result = await planner.generate_plan("test")
    assert result == []

    # 5. Unknown skill
    mock_plan = {
        "goal": "Test", "risk": "low", "steps": [{"description": "desc", "skill": "unknown"}]
    }
    mock_llm.chat_completion = AsyncMock(return_value=json.dumps(mock_plan))
    result = await planner.generate_plan("test")
    assert result == []

    # 6. Invalid skill_kwargs
    mock_skills.has_skill.return_value = True
    mock_plan = {
        "goal": "Test", "risk": "low", "steps": [{"description": "desc", "skill": "known", "skill_kwargs": "not_a_dict"}]
    }
    mock_llm.chat_completion = AsyncMock(return_value=json.dumps(mock_plan))
    result = await planner.generate_plan("test")
    assert result == []

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

@pytest.mark.asyncio
async def test_consciousness_plan_max_steps():
    from magda_agent.consciousness.core import Consciousness
    from magda_agent.emotions.engine import EmotionalEngine
    from magda_agent.memory.storage import MemorySystem

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.get_system_prompt.return_value = "System prompt"
    mock_llm.chat_completion = AsyncMock(return_value="Final LLM Response")

    mock_skills = MagicMock(spec=SkillRegistry)
    mock_skills.execute_skill.return_value = "Mocked result"
    mock_skills.get_skills_summary.return_value = "Available Skills..."

    real_planner = Planner(llm=mock_llm, skills=mock_skills)
    real_planner.generate_plan = AsyncMock()

    # set up a plan with 6 steps
    real_planner.current_plan = [{"description": f"Step {i}", "skill": None, "skill_kwargs": None} for i in range(6)]

    emotions = EmotionalEngine()
    memory = MemorySystem()

    consciousness = Consciousness(
        llm=mock_llm,
        emotions=emotions,
        memory=memory,
        skills=mock_skills,
        planner=real_planner
    )

    await consciousness.process_input("Help me test max steps")

    # verify only 5 steps were completed
    assert len(real_planner.completed_steps) == 5
    assert len(real_planner.current_plan) == 0

@pytest.mark.asyncio
async def test_consciousness_plan_timeout():
    from magda_agent.consciousness.core import Consciousness
    from magda_agent.emotions.engine import EmotionalEngine
    from magda_agent.memory.storage import MemorySystem
    import asyncio
    from unittest.mock import patch

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.get_system_prompt.return_value = "System prompt"
    mock_llm.chat_completion = AsyncMock(return_value="Final LLM Response")

    mock_skills = MagicMock(spec=SkillRegistry)
    mock_skills.get_skills_summary.return_value = "Available Skills..."

    real_planner = Planner(llm=mock_llm, skills=mock_skills)
    real_planner.generate_plan = AsyncMock()

    # set up a plan with 1 skill step
    real_planner.current_plan = [{"description": "Step 1", "skill": "slow_skill", "skill_kwargs": None}]

    emotions = EmotionalEngine()
    memory = MemorySystem()

    consciousness = Consciousness(
        llm=mock_llm,
        emotions=emotions,
        memory=memory,
        skills=mock_skills,
        planner=real_planner
    )

    with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
        await consciousness.process_input("Help me test timeout")

    # verify 1 step completed with timeout error, and pending plan is cleared
    assert len(real_planner.completed_steps) == 1
    assert "timed out after" in real_planner.completed_steps[0]["result"]
    assert len(real_planner.current_plan) == 0

def test_get_state_summary():
    mock_llm = MagicMock(spec=LLMClient)
    mock_skills = MagicMock(spec=SkillRegistry)
    planner = Planner(llm=mock_llm, skills=mock_skills)

    planner.current_goal = "Find answers"
    planner.current_risk = "low"
    planner.current_constraints = ["fast"]
    planner.current_plan = [{"description": "Pending step", "skill": None}]
    planner.completed_steps = [{"description": "Done step", "skill": "search", "result": "Found it"}]

    summary = planner.get_state_summary()

    assert "Goal: Find answers" in summary
    assert "Risk: low" in summary
    assert "Constraints: fast" in summary
    assert "Pending step" in summary
    assert "Done step" in summary
    assert "Found it" in summary
