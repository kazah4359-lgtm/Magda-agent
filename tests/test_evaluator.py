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
    mock = MagicMock(spec=MemorySystem)
    mock.add_memory = AsyncMock()
    return mock

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

@pytest.mark.asyncio
async def test_evaluate_response_retry_success(evaluator, mock_llm_client):
    # First response is invalid JSON, second response is valid JSON
    invalid_json_response = "this is not valid json"
    valid_json_response = '''{
      "usefulness": 9,
      "accuracy": 9,
      "completeness": 9,
      "emotional_adequacy": 9,
      "average_score": 9.0,
      "feedback": "Retry success"
    }'''
    mock_llm_client.chat_completion.side_effect = [invalid_json_response, valid_json_response]

    result = await evaluator.evaluate_response("Test", "Response")

    assert result is not None
    assert result["average_score"] == 9.0
    assert result["feedback"] == "Retry success"
    assert mock_llm_client.chat_completion.call_count == 2

@pytest.mark.asyncio
async def test_evaluate_response_retry_failure(evaluator, mock_llm_client):
    # Always return invalid JSON
    invalid_json_response = "still not valid json"
    mock_llm_client.chat_completion.side_effect = [invalid_json_response, invalid_json_response, invalid_json_response]

    result = await evaluator.evaluate_response("Test", "Response")

    assert result is None
    assert mock_llm_client.chat_completion.call_count == 3


@pytest.mark.asyncio
async def test_evaluator_agent_subagent_integration():
    from magda_agent.agents.evaluator import EvaluatorAgent
    from magda_agent.llm_client import LLMClient

    mock_llm = AsyncMock(spec=LLMClient)

    agent = EvaluatorAgent(llm=mock_llm)

    # Mock the sub_agent
    agent.sub_agent = AsyncMock()
    agent.sub_agent.execute.return_value = '{"score": 9, "approved": true, "feedback": "Looks good", "checklist_status": {"criteria1": "pass"}}'

    result = await agent.evaluate_output("print('hello')", "Write hello world", {"acceptance": ["Must use print"]})

    assert result["score"] == 9
    assert result["approved"] is True
    assert result["feedback"] == "Looks good"
    agent.sub_agent.execute.assert_called_once()

    # Verify execute arguments
    args, kwargs = agent.sub_agent.execute.call_args
    assert "task" in kwargs
    assert "context" in kwargs
    assert "Write hello world" in kwargs["context"]
    assert "Must use print" in kwargs["context"]
    assert "print('hello')" in kwargs["context"]
