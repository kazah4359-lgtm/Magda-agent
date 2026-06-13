import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from magda_agent.agents.generator_agent import GeneratorAgent
from magda_agent.safety.guardrails import RealtimeGuardrail, FallbackStrategy
from magda_agent.safety.policy import PolicyLayer

@pytest.mark.asyncio
async def test_guardrail_allows_legit_action():
    # Mock dependencies
    llm = MagicMock()
    skills = MagicMock()
    planner = MagicMock()
    policy = MagicMock(spec=PolicyLayer)

    # Setup policy to allow
    policy.evaluate.return_value = (True, "Allowed")

    guardrail = RealtimeGuardrail(policy_layer=policy)
    agent = GeneratorAgent(llm=llm, skills=skills, planner=planner, guardrail=guardrail)

    # Setup planner with a simple step
    step = {"id": "s1", "skill": "test_skill", "description": "test", "dependencies": []}
    planner.get_current_plan.return_value = [step]

    # Mock skill execution
    skills.execute_skill.return_value = "Success"

    # Let's mock the planner more realistically for the loop
    current_plan = [step]
    def mock_get_current_plan(user_id=None):
        return current_plan

    def mock_get_executable_steps(user_id=None):
        return current_plan

    def mock_mark_id_completed(step_id, result, user_id=None):
        nonlocal current_plan
        for i, s in enumerate(current_plan):
            if s["id"] == step_id:
                completed_step = current_plan.pop(i)
                completed_step['result'] = result
                planner.get_completed_steps.return_value.append(completed_step)
                return

    planner.get_current_plan.side_effect = mock_get_current_plan
    planner.get_executable_steps.side_effect = mock_get_executable_steps
    planner.mark_step_id_completed.side_effect = mock_mark_id_completed
    planner.get_completed_steps.return_value = []

    result = await agent.execute_plan("user input")

    assert "Success" in result
    assert "test_skill" in result
    policy.evaluate.assert_called_with("test_skill")

@pytest.mark.asyncio
async def test_guardrail_stops_on_violation():
    # Mock dependencies
    llm = MagicMock()
    skills = MagicMock()
    planner = MagicMock()
    policy = MagicMock(spec=PolicyLayer)

    # Setup policy to DENY with STOP_EXECUTION
    policy.evaluate.return_value = (False, "Policy Violation")

    guardrail = RealtimeGuardrail(policy_layer=policy, default_strategy=FallbackStrategy.STOP_EXECUTION)
    agent = GeneratorAgent(llm=llm, skills=skills, planner=planner, guardrail=guardrail)

    # Setup planner
    step = {"id": "s1", "skill": "dangerous_skill", "description": "dangerous", "dependencies": []}
    current_plan = [step]
    planner.get_current_plan.side_effect = lambda user_id=None: current_plan
    planner.get_executable_steps.side_effect = lambda user_id=None: current_plan
    planner.completed_steps = []

    def mock_mark_id_completed(step_id, result, user_id=None):
        nonlocal current_plan
        for i, s in enumerate(current_plan):
            if s["id"] == step_id:
                completed_step = current_plan.pop(i)
                completed_step['result'] = result
                planner.get_completed_steps.return_value.append(completed_step)
                return

    planner.mark_step_id_completed.side_effect = mock_mark_id_completed
    planner.get_completed_steps.return_value = []

    result = await agent.execute_plan("user input")

    assert "Guardrail Fallback (STOP): Policy Violation" in result
    assert "dangerous_skill" in result
    skills.execute_skill.assert_not_called()
    planner.clear_pending_plan.assert_called_once()

@pytest.mark.asyncio
async def test_guardrail_review_required():
    # Mock dependencies
    llm = MagicMock()
    skills = MagicMock()
    planner = MagicMock()
    policy = MagicMock(spec=PolicyLayer)

    # Setup policy to DENY with REQUEST_REVIEW
    policy.evaluate.return_value = (False, "Review needed")

    guardrail = RealtimeGuardrail(policy_layer=policy, default_strategy=FallbackStrategy.REQUEST_REVIEW)
    agent = GeneratorAgent(llm=llm, skills=skills, planner=planner, guardrail=guardrail)

    # Setup planner
    step = {"id": "s1", "skill": "sketchy_skill", "description": "sketchy", "dependencies": []}
    current_plan = [step]
    planner.get_current_plan.side_effect = lambda user_id=None: current_plan
    planner.get_executable_steps.side_effect = lambda user_id=None: current_plan
    planner.completed_steps = []

    def mock_mark_id_completed(step_id, result, user_id=None):
        nonlocal current_plan
        for i, s in enumerate(current_plan):
            if s["id"] == step_id:
                completed_step = current_plan.pop(i)
                completed_step['result'] = result
                planner.get_completed_steps.return_value.append(completed_step)
                return

    planner.mark_step_id_completed.side_effect = mock_mark_id_completed
    planner.get_completed_steps.return_value = []

    result = await agent.execute_plan("user input")

    assert "Guardrail Fallback (REVIEW REQUIRED): Review needed" in result
    skills.execute_skill.assert_not_called()
    planner.clear_pending_plan.assert_called_once()
