import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.agents.triad_coordinator import TriadCoordinator
from magda_agent.agents.planner_agent import PlannerAgent
from magda_agent.agents.generator_agent import GeneratorAgent
from magda_agent.agents.evaluator_agent import EvaluatorAgent

@pytest.mark.asyncio
async def test_triad_coordinator():
    planner = AsyncMock(spec=PlannerAgent)
    generator = AsyncMock(spec=GeneratorAgent)
    evaluator = AsyncMock(spec=EvaluatorAgent)

    generator.execute_plan.return_value = "plan string"
    generator.generate_response.return_value = "final response"

    coordinator = TriadCoordinator(planner, generator, evaluator)

    message_builder = MagicMock(return_value=[{"role": "system", "content": "hello"}])
    pre_hook = MagicMock()

    res = await coordinator.coordinate(
        "hello",
        user_id="123",
        message_builder=message_builder,
        pre_generation_hook=pre_hook,
        policies=["policy1"]
    )

    assert res == "final response"
    planner.plan.assert_called_once_with("hello", user_id="123", mental_state=None)
    generator.execute_plan.assert_called_once_with("hello", user_id="123")
    message_builder.assert_called_once_with("plan string")
    pre_hook.assert_called_once()
    generator.generate_response.assert_called_once_with([{"role": "system", "content": "hello"}])
    evaluator.evaluate.assert_called_once_with("hello", "final response", user_id="123", policies=["policy1"])
