import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.emotions.mental_states import MentalStates, MentalState
from magda_agent.planning.planner import Planner
from magda_agent.llm_client import LLMClient
from magda_agent.skills.registry import SkillRegistry

def test_mental_states_biases():
    ms = MentalStates()
    user_id = 123

    # Initial state
    state = ms._get_state(user_id)
    assert state.optimism == 0.5
    assert state.overconfidence == 0.0

    # Success update
    ms.update_from_action_result(success=True, user_id=user_id)
    # optimism: 0.5 -> 0.55
    # overconfidence: 0.0 -> 0.05
    assert abs(state.optimism - 0.55) < 1e-9
    assert abs(state.overconfidence - 0.05) < 1e-9

    # Failure update
    ms.update_from_action_result(success=False, user_id=user_id)
    # optimism: 0.55 -> 0.45
    # overconfidence: 0.05 -> 0.0
    assert abs(state.optimism - 0.45) < 1e-9
    assert abs(state.overconfidence - 0.0) < 1e-9

    # Manual modifier
    ms.apply_bias_modifier(optimism_mod=0.5, overconfidence_mod=0.8, user_id=user_id)
    # optimism: 0.45 + 0.5 = 0.95
    # overconfidence: 0.0 + 0.8 = 0.8
    assert abs(state.optimism - 0.95) < 1e-9
    assert abs(state.overconfidence - 0.8) < 1e-9

@pytest.mark.asyncio
async def test_planner_bias_modulation():
    llm = MagicMock(spec=LLMClient)
    llm.chat_completion = AsyncMock(return_value='{"goal": "test", "constraints": [], "risk": "low", "steps": [], "acceptance": []}')
    skills = MagicMock(spec=SkillRegistry)
    skills.get_skills_summary.return_value = "No skills"

    planner = Planner(llm=llm, skills=skills)

    # High optimism mental state
    high_bias = MentalState(optimism=0.9, overconfidence=0.9)
    await planner.generate_plan("Do something", mental_state=high_bias)

    # Verify LLM was called with bias instructions
    args, _ = llm.chat_completion.call_args
    system_prompt = args[0][0]["content"]
    assert "optimistic" in system_prompt
    assert "very confident in your abilities" in system_prompt

    # Low optimism
    low_bias = MentalState(optimism=0.1)
    await planner.generate_plan("Do something", mental_state=low_bias)
    args, _ = llm.chat_completion.call_args
    system_prompt = args[0][0]["content"]
    assert "pessimistic" in system_prompt
    assert "cautious" in system_prompt.lower()
