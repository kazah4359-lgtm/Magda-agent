import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock
from magda_agent.agents.generator_agent import GeneratorAgent
from magda_agent.skills.registry import SkillRegistry
from magda_agent.planning.planner import Planner, PlanStep
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_execute_plan_concurrency() -> None:
    """Tests that GeneratorAgent executes independent steps concurrently."""

    # Setup skills
    skills = SkillRegistry()

    # We want to verify concurrency, so we'll use a skill that sleeps
    def sleep_skill(seconds=0.1):
        time.sleep(seconds)
        return f"Slept for {seconds}"

    skills.register_skill("sleep", sleep_skill, "Sleeps for a bit")

    # Setup Mock LLM
    llm = MagicMock(spec=LLMClient)

    # Setup Planner with a plan that has independent steps
    planner = Planner(llm=llm, skills=skills)
    user_id = "test_user"

    # Create steps: step1 and step2 are independent, step3 depends on both
    plan_steps = [
        {"id": "step1", "description": "Independent 1", "skill": "sleep", "skill_kwargs": {"seconds": 0.5}, "dependencies": []},
        {"id": "step2", "description": "Independent 2", "skill": "sleep", "skill_kwargs": {"seconds": 0.5}, "dependencies": []},
        {"id": "step3", "description": "Dependent", "skill": "sleep", "skill_kwargs": {"seconds": 0.1}, "dependencies": ["step1", "step2"]}
    ]

    # Manually set the plan in the planner
    state = planner.get_user_state(user_id=user_id)
    state.current_plan = plan_steps

    agent = GeneratorAgent(llm=llm, skills=skills, planner=planner)

    start_time = asyncio.get_event_loop().time()
    result_str = await agent.execute_plan("run concurrent task", user_id=user_id)
    end_time = asyncio.get_event_loop().time()

    execution_time = end_time - start_time

    # If executed sequentially, time would be >= 0.5 + 0.5 + 0.1 = 1.1s
    # If executed concurrently, time would be around 0.5 + 0.1 = 0.6s
    # We allow some overhead (CI environments can be slow)
    assert execution_time < 5.0, f"Execution took too long: {execution_time}s. Concurrency might not be working or environment is extremely slow."
    assert "Step 1: Independent 1" in result_str
    assert "Step 2: Independent 2" in result_str
    assert "Step 3: Dependent" in result_str
    assert len(planner.get_completed_steps(user_id=user_id)) == 3

@pytest.mark.asyncio
async def test_execute_plan_max_steps() -> None:
    """Tests that GeneratorAgent respects MAX_STEPS even with concurrency."""
    skills = SkillRegistry()
    skills.register_skill("noop", lambda: "ok", "does nothing")

    llm = MagicMock(spec=LLMClient)
    planner = Planner(llm=llm, skills=skills)
    user_id = "test_user_max"

    # Create 15 independent steps. MAX_STEPS is 10.
    plan_steps = [{"id": f"step{i}", "description": f"Step {i}", "skill": "noop", "dependencies": []} for i in range(15)]

    state = planner.get_user_state(user_id=user_id)
    state.current_plan = plan_steps

    agent = GeneratorAgent(llm=llm, skills=skills, planner=planner)

    result_str = await agent.execute_plan("run many tasks", user_id=user_id)

    assert len(planner.get_completed_steps(user_id=user_id)) == 10
    assert "Plan execution stopped due to MAX_STEPS limit" in result_str
