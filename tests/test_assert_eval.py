import pytest
import json
from unittest.mock import AsyncMock
from magda_agent.evaluation.assert_eval import AssertPolicyEvaluator
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_llm_client():
    mock = AsyncMock(spec=LLMClient)
    return mock

@pytest.fixture
def evaluator(mock_llm_client):
    return AssertPolicyEvaluator(llm=mock_llm_client)

@pytest.mark.asyncio
async def test_evaluate_response_compliant(evaluator, mock_llm_client):
    mock_json_response = '''
    {
      "is_compliant": true,
      "violations": [],
      "score": 1.0
    }
    '''
    mock_llm_client.chat_completion.return_value = mock_json_response
    policies = ["Must be helpful"]
    response = "I can help with that."
    result = await evaluator.evaluate_response(response, policies)
    assert result["is_compliant"] is True
    mock_llm_client.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_evaluate_response_non_compliant(evaluator, mock_llm_client):
    mock_json_response = '''
    {
      "is_compliant": false,
      "violations": ["Must be polite"],
      "score": 0.0
    }
    '''
    mock_llm_client.chat_completion.return_value = mock_json_response
    policies = ["Must be polite"]
    response = "Do it yourself."
    result = await evaluator.evaluate_response(response, policies)
    assert result["is_compliant"] is False
    assert result["violations"] == ["Must be polite"]

@pytest.mark.asyncio
async def test_evaluate_response_failure(evaluator, mock_llm_client):
    mock_llm_client.chat_completion.side_effect = Exception("API Error")
    policies = ["Policy"]
    response = "Response"
    result = await evaluator.evaluate_response(response, policies)
    assert result["is_compliant"] is False
    assert result["score"] == 0.0
