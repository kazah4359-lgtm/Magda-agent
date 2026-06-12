import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock
from magda_agent.agents.generator_agent import GeneratorAgent
from magda_agent.skills.registry import SkillRegistry
from magda_agent.planning.planner import Planner
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_generator_agent_concurrency():
    mock_llm = MagicMock(spec=LLMClient)
    mock_skills = MagicMock(spec=SkillRegistry)

    # Define a slow skill
    # We use a wrapper because SkillRegistry.execute_skill will be mocked
    # and we want to control what it returns based on name and kwargs.
    def execute_skill_mock(name, delay=0.1, result="ok"):
        time.sleep(delay)
        return result

    mock_skills.execute_skill.side_effect = execute_skill_mock
    mock_skills.has_skill.return_value = True

    planner = Planner(llm=mock_llm, skills=mock_skills)

    # Create a plan with 3 independent steps
    planner.current_plan = [
        {"id": "step1", "description": "Step 1", "skill": "slow_skill", "skill_kwargs": {"delay": 0.5, "result": "res1"}, "dependencies": []},
        {"id": "step2", "description": "Step 2", "skill": "slow_skill", "skill_kwargs": {"delay": 0.5, "result": "res2"}, "dependencies": []},
        {"id": "step3", "description": "Step 3", "skill": "slow_skill", "skill_kwargs": {"delay": 0.5, "result": "res3"}, "dependencies": []},
    ]
    planner.current_goal = "Test concurrency"

    agent = GeneratorAgent(llm=mock_llm, skills=mock_skills, planner=planner)

    start_time = asyncio.get_event_loop().time()
    await agent.execute_plan("Test input")
    end_time = asyncio.get_event_loop().time()

    duration = end_time - start_time

    # If concurrent, it should take ~0.5s + overhead.
    # If sequential, it should take ~1.5s.
    # We assert it's less than 1.0s to be safe.
    assert duration < 1.0
    assert len(planner.completed_steps) == 3
    results = {s["id"]: s["result"] for s in planner.completed_steps}
    assert results["step1"] == "res1"
    assert results["step2"] == "res2"
    assert results["step3"] == "res3"

@pytest.mark.asyncio
async def test_generator_agent_dependency_respect():
    mock_llm = MagicMock(spec=LLMClient)
    mock_skills = MagicMock(spec=SkillRegistry)

    mock_skills.execute_skill.side_effect = lambda name, result="ok": result
    mock_skills.has_skill.return_value = True

    planner = Planner(llm=mock_llm, skills=mock_skills)

    # Step 1 -> Step 2
    planner.current_plan = [
        {"id": "step1", "description": "Step 1", "skill": "skill", "skill_kwargs": {"result": "res1"}, "dependencies": []},
        {"id": "step2", "description": "Step 2", "skill": "skill", "skill_kwargs": {"result": "res2"}, "dependencies": ["step1"]},
    ]
    planner.current_goal = "Test dependencies"

    agent = GeneratorAgent(llm=mock_llm, skills=mock_skills, planner=planner)

    await agent.execute_plan("Test input")

    assert len(planner.completed_steps) == 2
    assert planner.completed_steps[0]["id"] == "step1"
    assert planner.completed_steps[1]["id"] == "step2"

@pytest.mark.asyncio
async def test_generator_agent_partial_concurrency():
    mock_llm = MagicMock(spec=LLMClient)
    mock_skills = MagicMock(spec=SkillRegistry)

    def execute_skill_mock(name, delay=0.1, result="ok"):
        time.sleep(delay)
        return result

    mock_skills.execute_skill.side_effect = execute_skill_mock
    mock_skills.has_skill.return_value = True

    planner = Planner(llm=mock_llm, skills=mock_skills)

    # Step 1 & Step 2 are independent. Step 3 depends on Step 1.
    planner.current_plan = [
        {"id": "step1", "description": "Step 1", "skill": "slow_skill", "skill_kwargs": {"delay": 0.5, "result": "res1"}, "dependencies": []},
        {"id": "step2", "description": "Step 2", "skill": "slow_skill", "skill_kwargs": {"delay": 0.5, "result": "res2"}, "dependencies": []},
        {"id": "step3", "description": "Step 3", "skill": "slow_skill", "skill_kwargs": {"delay": 0.5, "result": "res3"}, "dependencies": ["step1"]},
    ]
    planner.current_goal = "Test partial concurrency"

    agent = GeneratorAgent(llm=mock_llm, skills=mock_skills, planner=planner)

    start_time = asyncio.get_event_loop().time()
    await agent.execute_plan("Test input")
    end_time = asyncio.get_event_loop().time()

    duration = end_time - start_time

    # Execution:
    # Round 1: step1, step2 (parallel) -> ~0.5s
    # Round 2: step3 -> ~0.5s
    # Total: ~1.0s.
    # Sequential would be 1.5s.

    assert 0.8 < duration < 1.3
    assert len(planner.completed_steps) == 3
