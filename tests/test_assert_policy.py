import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from magda_agent.evals.assert_policy import AssertPolicyEvaluator
from magda_agent.llm_client import LLMClient
from magda_agent.memory.storage import MemorySystem

@pytest.fixture
def mock_llm_client():
    mock = AsyncMock(spec=LLMClient)
    # Default successful mock response
    mock_json_response = '''
    ```json
    {
      "policy_adherence_score": 9.0,
      "violated_policies": [],
      "feedback": "Complies well with the provided policies"
    }
    ```
    '''
    mock.chat_completion.return_value = mock_json_response
    return mock

@pytest.fixture
def mock_memory_system():
    mock = MagicMock(spec=MemorySystem)
    mock.add_memory = AsyncMock()
    return mock

@pytest.fixture
def evaluator(mock_llm_client, mock_memory_system):
    return AssertPolicyEvaluator(llm=mock_llm_client, memory=mock_memory_system)

@pytest.mark.asyncio
async def test_evaluate_response_with_policy_success(evaluator, mock_llm_client, mock_memory_system) -> None:
    user_input = "Hello"
    agent_response = "Hi there!"
    policies = ["Be polite"]

    result = await evaluator.evaluate_response_with_policy(user_input, agent_response, policies)

    # Assert LLM was called
    mock_llm_client.chat_completion.assert_called_once()

    # Assert policies are in the prompt
    call_args = mock_llm_client.chat_completion.call_args[0][0]
    assert "- Be polite" in call_args[0]["content"]

    # Assert memory was added
    mock_memory_system.add_memory.assert_called_once()
    mem_call_args = mock_memory_system.add_memory.call_args[1]
    assert "Score: 9.0" in mem_call_args["content"]
    assert mem_call_args["tags"] == ["evaluation", "metacognition", "policy"]

    # Assert result
    assert result is not None
    assert result["policy_adherence_score"] == 9.0
    assert result["feedback"] == "Complies well with the provided policies"

    # Assert state updated
    assert evaluator.last_evaluation == result

@pytest.mark.asyncio
async def test_evaluate_response_with_policy_no_markdown(evaluator, mock_llm_client) -> None:
    mock_json_response = '''{
      "policy_adherence_score": 5.0,
      "violated_policies": ["Do not use exclamation marks"],
      "feedback": "Failed to adhere to policy"
    }'''
    mock_llm_client.chat_completion.return_value = mock_json_response

    result = await evaluator.evaluate_response_with_policy("Test", "Response!", ["Do not use exclamation marks"])
    assert result is not None
    assert result["policy_adherence_score"] == 5.0

@pytest.mark.asyncio
async def test_get_feedback_for_prompt_high_score(evaluator) -> None:
    evaluator.last_evaluation = {"policy_adherence_score": 8.5, "feedback": "Great"}
    feedback = evaluator.get_feedback_for_prompt()
    assert feedback == "" # Should be empty for high scores

@pytest.mark.asyncio
async def test_get_feedback_for_prompt_low_score(evaluator) -> None:
    evaluator.last_evaluation = {"policy_adherence_score": 4.5, "violated_policies": ["No swearing"], "feedback": "Policy violation"}
    feedback = evaluator.get_feedback_for_prompt()
    assert "low policy adherence score (4.5/10)" in feedback
    assert "Violated policies: No swearing" in feedback
    assert "Policy violation" in feedback

@pytest.mark.asyncio
async def test_get_feedback_for_prompt_no_evaluation(evaluator) -> None:
    feedback = evaluator.get_feedback_for_prompt()
    assert feedback == ""

@pytest.mark.asyncio
async def test_evaluate_response_with_policy_exception(evaluator, mock_llm_client) -> None:
    mock_llm_client.chat_completion.side_effect = Exception("API Error")
    result = await evaluator.evaluate_response_with_policy("Test", "Response", [])
    assert result is None

@pytest.mark.asyncio
async def test_evaluate_response_with_policy_retry_success(evaluator, mock_llm_client) -> None:
    # First response is invalid JSON, second response is valid JSON
    invalid_json_response = "this is not valid json"
    valid_json_response = '''{
      "policy_adherence_score": 9.0,
      "violated_policies": [],
      "feedback": "Retry success"
    }'''
    mock_llm_client.chat_completion.side_effect = [invalid_json_response, valid_json_response]

    result = await evaluator.evaluate_response_with_policy("Test", "Response", [])

    assert result is not None
    assert result["policy_adherence_score"] == 9.0
    assert result["feedback"] == "Retry success"
    assert mock_llm_client.chat_completion.call_count == 2

@pytest.mark.asyncio
async def test_evaluate_response_with_policy_retry_failure(evaluator, mock_llm_client) -> None:
    # Always return invalid JSON
    invalid_json_response = "still not valid json"
    mock_llm_client.chat_completion.side_effect = [invalid_json_response, invalid_json_response, invalid_json_response]

    result = await evaluator.evaluate_response_with_policy("Test", "Response", [])

    assert result is None
    assert mock_llm_client.chat_completion.call_count == 3
