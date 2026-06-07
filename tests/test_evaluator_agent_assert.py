import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.agents.evaluator_agent import EvaluatorAgent
from magda_agent.metacognition.evaluator import Evaluator
from magda_agent.metacognition.assert_evaluator import AssertEvaluator

@pytest.fixture
def mock_evaluator():
    evaluator = MagicMock(spec=Evaluator)
    evaluator.evaluate_response = AsyncMock()
    evaluator.last_evaluation = {"average_score": 8.0}
    return evaluator

@pytest.fixture
def mock_assert_evaluator():
    assert_evaluator = MagicMock(spec=AssertEvaluator)
    assert_evaluator.evaluate_response_with_policy = AsyncMock()
    return assert_evaluator

@pytest.mark.asyncio
async def test_evaluator_agent_passes_policies(mock_evaluator, mock_assert_evaluator):
    agent = EvaluatorAgent(evaluator=mock_evaluator, assert_evaluator=mock_assert_evaluator)

    user_input = "Hello"
    response = "Hi"
    policies = ["Be polite"]

    await agent.evaluate(user_input, response, policies=policies)

    mock_evaluator.evaluate_response.assert_called_once_with(user_input, response)
    mock_assert_evaluator.evaluate_response_with_policy.assert_called_once_with(user_input, response, policies)

@pytest.mark.asyncio
async def test_evaluator_agent_no_policies(mock_evaluator, mock_assert_evaluator):
    agent = EvaluatorAgent(evaluator=mock_evaluator, assert_evaluator=mock_assert_evaluator)

    user_input = "Hello"
    response = "Hi"

    await agent.evaluate(user_input, response)

    mock_evaluator.evaluate_response.assert_called_once_with(user_input, response)
    mock_assert_evaluator.evaluate_response_with_policy.assert_not_called()

@pytest.mark.asyncio
async def test_evaluator_agent_no_assert_evaluator(mock_evaluator):
    agent = EvaluatorAgent(evaluator=mock_evaluator)

    user_input = "Hello"
    response = "Hi"
    policies = ["Be polite"]

    await agent.evaluate(user_input, response, policies=policies)

    mock_evaluator.evaluate_response.assert_called_once_with(user_input, response)

@pytest.mark.asyncio
async def test_evaluator_agent_only_assert_evaluator(mock_assert_evaluator):
    agent = EvaluatorAgent(assert_evaluator=mock_assert_evaluator)

    user_input = "Hello"
    response = "Hi"
    policies = ["Be polite"]

    await agent.evaluate(user_input, response, policies=policies)

    mock_assert_evaluator.evaluate_response_with_policy.assert_called_once_with(user_input, response, policies)
