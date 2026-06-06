import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.agents.planner_agent import PlannerAgent
from magda_agent.agents.generator_agent import GeneratorAgent
from magda_agent.agents.evaluator_agent import EvaluatorAgent

@pytest.mark.asyncio
async def test_planner_agent():
    mock_planner = MagicMock()
    mock_planner.get_current_plan.side_effect = [[], [{"step": 1}]]
    mock_planner.generate_plan = AsyncMock()

    agent = PlannerAgent(planner=mock_planner)
    plan = await agent.plan("test input")

    assert plan == [{"step": 1}]
    mock_planner.generate_plan.assert_called_once_with("test input", user_id=None)

@pytest.mark.asyncio
async def test_generator_agent():
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value="generated response")

    mock_planner = MagicMock()

    def mock_get_current_plan():
        if getattr(mock_planner, 'cleared', False):
            return []
        return [{"skill": "test_skill", "skill_kwargs": {}}]
    mock_planner.cleared = False
    mock_planner.get_current_plan.side_effect = mock_get_current_plan

    mock_planner.completed_steps = [{"description": "step1", "skill": "test_skill", "result": "success"}]
    def mock_mark_completed(*args):
        mock_planner.cleared = True
    mock_planner.mark_step_completed.side_effect = mock_mark_completed

    mock_skills = MagicMock()
    mock_skills.execute_skill = MagicMock(return_value="success")

    agent = GeneratorAgent(
        llm=mock_llm,
        skills=mock_skills,
        planner=mock_planner
    )

    plan_str = await agent.execute_plan("test input")
    assert "Executed Plan Results" in plan_str
    assert "success" in plan_str

    response = await agent.generate_response([{"role": "user", "content": "test input"}])
    assert response == "generated response"
    mock_llm.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_evaluator_agent():
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_response = AsyncMock()
    mock_evaluator.last_evaluation = {"average_score": 9.0}

    mock_confidence = MagicMock()
    mock_confidence.last_confidence = 0.8
    mock_confidence.track_calibration = MagicMock()

    agent = EvaluatorAgent(
        evaluator=mock_evaluator,
        confidence_calibrator=mock_confidence
    )

    await agent.evaluate("test input", "response")
    mock_evaluator.evaluate_response.assert_called_once_with("test input", "response")
    mock_confidence.track_calibration.assert_called_once_with(0.8, 9.0)
