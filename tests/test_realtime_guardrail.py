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
    planner.get_current_plan.return_value = [{"skill": "test_skill", "description": "test"}]

    # Mock skill execution
    skills.execute_skill.return_value = "Success"

    # We need to mock mark_step_completed to avoid issues if it tries to pop from the list
    # Actually, execute_plan uses self.planner.get_current_plan()[0] and then marks it completed.
    # In the real Planner, mark_step_completed moves the step.

    # Let's mock the planner more realistically for the loop
    current_plan = [{"skill": "test_skill", "description": "test"}]
    def mock_get_current_plan():
        return current_plan

    def mock_mark_completed(index, result):
        nonlocal current_plan
        step = current_plan.pop(index)
        step['result'] = result
        agent.planner.completed_steps.append(step)

    planner.get_current_plan.side_effect = mock_get_current_plan
    planner.mark_step_completed.side_effect = mock_mark_completed
    planner.completed_steps = []

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
    current_plan = [{"skill": "dangerous_skill", "description": "dangerous"}]
    planner.get_current_plan.side_effect = lambda: current_plan
    planner.completed_steps = []

    def mock_mark_completed(index, result):
        nonlocal current_plan
        step = current_plan.pop(index)
        step['result'] = result
        agent.planner.completed_steps.append(step)

    planner.mark_step_completed.side_effect = mock_mark_completed

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
    current_plan = [{"skill": "sketchy_skill", "description": "sketchy"}]
    planner.get_current_plan.side_effect = lambda: current_plan
    planner.completed_steps = []

    def mock_mark_completed(index, result):
        nonlocal current_plan
        step = current_plan.pop(index)
        step['result'] = result
        agent.planner.completed_steps.append(step)

    planner.mark_step_completed.side_effect = mock_mark_completed

    result = await agent.execute_plan("user input")

    assert "Guardrail Fallback (REVIEW REQUIRED): Review needed" in result
    skills.execute_skill.assert_not_called()
    planner.clear_pending_plan.assert_called_once()
