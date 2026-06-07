import pytest
import json
from unittest.mock import AsyncMock

from magda_agent.safety.assert_eval import AssertWorkflowEvaluator
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_llm_client():
    mock = AsyncMock(spec=LLMClient)
    return mock

@pytest.fixture
def evaluator(mock_llm_client):
    return AssertWorkflowEvaluator(llm=mock_llm_client)

@pytest.mark.asyncio
async def test_evaluate_workflow_safe(evaluator, mock_llm_client):
    mock_json_response = '''
    {
      "is_safe": true,
      "violated_policies": [],
      "reason": "The workflow action complies with all policies."
    }
    '''
    mock_llm_client.chat_completion.return_value = mock_json_response

    workflow_data = {"action": "read_file", "path": "docs/readme.md"}
    policies = ["Do not access secrets"]

    is_safe = await evaluator.evaluate_workflow(workflow_data, policies)
    assert is_safe is True
    mock_llm_client.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_evaluate_workflow_unsafe(evaluator, mock_llm_client):
    mock_json_response = '''
    {
      "is_safe": false,
      "violated_policies": ["Do not access secrets"],
      "reason": "The workflow attempts to read a .env file which is forbidden."
    }
    '''
    mock_llm_client.chat_completion.return_value = mock_json_response

    workflow_data = {"action": "read_file", "path": ".env"}
    policies = ["Do not access secrets"]

    is_safe = await evaluator.evaluate_workflow(workflow_data, policies)
    assert is_safe is False

@pytest.mark.asyncio
async def test_evaluate_workflow_invalid_json_retry(evaluator, mock_llm_client):
    invalid_json = "not json"
    valid_json = '''
    {
      "is_safe": true,
      "violated_policies": [],
      "reason": "Success after retry"
    }
    '''
    mock_llm_client.chat_completion.side_effect = [invalid_json, valid_json]

    workflow_data = {"action": "test"}
    policies = ["Test policy"]

    is_safe = await evaluator.evaluate_workflow(workflow_data, policies)
    assert is_safe is True
    assert mock_llm_client.chat_completion.call_count == 2

@pytest.mark.asyncio
async def test_evaluate_workflow_fail_closed_on_error(evaluator, mock_llm_client):
    mock_llm_client.chat_completion.side_effect = Exception("API down")

    workflow_data = {"action": "test"}
    policies = ["Test policy"]

    is_safe = await evaluator.evaluate_workflow(workflow_data, policies)
    # Evaluator should fail closed (return False) on exception
    assert is_safe is False
