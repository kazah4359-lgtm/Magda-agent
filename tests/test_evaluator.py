import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from magda_agent.metacognition.evaluator import Evaluator
from magda_agent.llm_client import LLMClient
from magda_agent.memory.storage import MemorySystem

@pytest.fixture
def mock_llm_client():
    mock = AsyncMock(spec=LLMClient)
    # Default successful mock response
    mock_json_response = '''
    ```json
    {
      "usefulness": 8,
      "accuracy": 9,
      "completeness": 7,
      "emotional_adequacy": 8,
      "average_score": 8.0,
      "feedback": "Good response"
    }
    ```
    '''
    mock.chat_completion.return_value = mock_json_response
    return mock

@pytest.fixture
def mock_memory_system():
    return MagicMock(spec=MemorySystem)

@pytest.fixture
def evaluator(mock_llm_client, mock_memory_system):
    return Evaluator(llm=mock_llm_client, memory=mock_memory_system)

@pytest.mark.asyncio
async def test_evaluate_response_success(evaluator, mock_llm_client, mock_memory_system):
    user_input = "Hello"
    agent_response = "Hi there!"

    result = await evaluator.evaluate_response(user_input, agent_response)

    # Assert LLM was called
    mock_llm_client.chat_completion.assert_called_once()

    # Assert memory was added
    mock_memory_system.add_memory.assert_called_once()
    call_args = mock_memory_system.add_memory.call_args[1]
    assert "Avg Score: 8.0" in call_args["content"]
    assert call_args["tags"] == ["evaluation", "metacognition"]

    # Assert result
    assert result is not None
    assert result["average_score"] == 8.0
    assert result["feedback"] == "Good response"

    # Assert state updated
    assert evaluator.last_evaluation == result

@pytest.mark.asyncio
async def test_evaluate_response_no_markdown(evaluator, mock_llm_client):
    mock_json_response = '''{
      "usefulness": 5,
      "accuracy": 5,
      "completeness": 5,
      "emotional_adequacy": 5,
      "average_score": 5.0,
      "feedback": "Average response"
    }'''
    mock_llm_client.chat_completion.return_value = mock_json_response

    result = await evaluator.evaluate_response("Test", "Response")
    assert result is not None
    assert result["average_score"] == 5.0

@pytest.mark.asyncio
async def test_get_feedback_for_prompt_high_score(evaluator):
    evaluator.last_evaluation = {"average_score": 8.5, "feedback": "Great"}
    feedback = evaluator.get_feedback_for_prompt()
    assert feedback == "" # Should be empty for high scores

@pytest.mark.asyncio
async def test_get_feedback_for_prompt_low_score(evaluator):
    evaluator.last_evaluation = {"average_score": 4.5, "feedback": "Poor response"}
    feedback = evaluator.get_feedback_for_prompt()
    assert "low evaluation score (4.5/10)" in feedback
    assert "Poor response" in feedback

@pytest.mark.asyncio
async def test_get_feedback_for_prompt_no_evaluation(evaluator):
    feedback = evaluator.get_feedback_for_prompt()
    assert feedback == ""

@pytest.mark.asyncio
async def test_evaluate_response_exception(evaluator, mock_llm_client):
    mock_llm_client.chat_completion.side_effect = Exception("API Error")
    result = await evaluator.evaluate_response("Test", "Response")
    assert result is None
