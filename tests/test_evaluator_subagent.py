import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from magda_agent.agents.evaluator_subagent import EvaluatorSubagent
from magda_agent.llm_client import LLMClient
from magda_agent.memory.storage import MemorySystem

@pytest.fixture
def mock_llm_client():
    return AsyncMock(spec=LLMClient)

@pytest.fixture
def mock_memory_system():
    mock = MagicMock(spec=MemorySystem)
    mock.add_memory = AsyncMock()
    return mock

@pytest.fixture
def evaluator_subagent(mock_llm_client, mock_memory_system):
    return EvaluatorSubagent(llm=mock_llm_client, memory=mock_memory_system)

@pytest.mark.asyncio
async def test_evaluate_response_success(evaluator_subagent, mock_memory_system):
    evaluator_subagent.sub_agent.execute = AsyncMock()
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
    evaluator_subagent.sub_agent.execute.return_value = mock_json_response

    user_input = "Hello"
    agent_response = "Hi there!"

    result = await evaluator_subagent.evaluate_response(user_input, agent_response)

    # Assert sub_agent execute was called
    evaluator_subagent.sub_agent.execute.assert_called_once()
    kwargs = evaluator_subagent.sub_agent.execute.call_args.kwargs
    assert "task" in kwargs
    assert "context" in kwargs
    assert "temperature" in kwargs

    # Assert memory was added
    mock_memory_system.add_memory.assert_called_once()
    call_args = mock_memory_system.add_memory.call_args[1]
    assert "Avg Score: 8.0" in call_args["content"]
    assert "evaluation" in call_args["tags"]
    assert "subagent" in call_args["tags"]

    # Assert result
    assert result is not None
    assert result["average_score"] == 8.0
    assert result["feedback"] == "Good response"

    # Assert state updated
    assert evaluator_subagent.last_evaluation == result

@pytest.mark.asyncio
async def test_evaluate_response_no_markdown(evaluator_subagent):
    evaluator_subagent.sub_agent.execute = AsyncMock()
    mock_json_response = '''{
      "usefulness": 5,
      "accuracy": 5,
      "completeness": 5,
      "emotional_adequacy": 5,
      "average_score": 5.0,
      "feedback": "Average response"
    }'''
    evaluator_subagent.sub_agent.execute.return_value = mock_json_response

    result = await evaluator_subagent.evaluate_response("Test", "Response")
    assert result is not None
    assert result["average_score"] == 5.0

@pytest.mark.asyncio
async def test_get_feedback_for_prompt_high_score(evaluator_subagent):
    evaluator_subagent.last_evaluation = {"average_score": 8.5, "feedback": "Great"}
    feedback = evaluator_subagent.get_feedback_for_prompt()
    assert feedback == ""

@pytest.mark.asyncio
async def test_get_feedback_for_prompt_low_score(evaluator_subagent):
    evaluator_subagent.last_evaluation = {"average_score": 4.5, "feedback": "Poor response"}
    feedback = evaluator_subagent.get_feedback_for_prompt()
    assert "low evaluation score (4.5/10)" in feedback
    assert "Poor response" in feedback

@pytest.mark.asyncio
async def test_get_feedback_for_prompt_no_evaluation(evaluator_subagent):
    feedback = evaluator_subagent.get_feedback_for_prompt()
    assert feedback == ""

@pytest.mark.asyncio
async def test_evaluate_response_exception(evaluator_subagent):
    evaluator_subagent.sub_agent.execute = AsyncMock(side_effect=Exception("API Error"))
    result = await evaluator_subagent.evaluate_response("Test", "Response")
    assert result is None

@pytest.mark.asyncio
async def test_evaluate_response_retry_success(evaluator_subagent):
    evaluator_subagent.sub_agent.execute = AsyncMock()
    invalid_json_response = "this is not valid json"
    valid_json_response = '''{
      "usefulness": 9,
      "accuracy": 9,
      "completeness": 9,
      "emotional_adequacy": 9,
      "average_score": 9.0,
      "feedback": "Retry success"
    }'''
    evaluator_subagent.sub_agent.execute.side_effect = [invalid_json_response, valid_json_response]

    result = await evaluator_subagent.evaluate_response("Test", "Response")

    assert result is not None
    assert result["average_score"] == 9.0
    assert result["feedback"] == "Retry success"
    assert evaluator_subagent.sub_agent.execute.call_count == 2

@pytest.mark.asyncio
async def test_evaluate_response_retry_failure(evaluator_subagent):
    evaluator_subagent.sub_agent.execute = AsyncMock()
    invalid_json_response = "still not valid json"
    evaluator_subagent.sub_agent.execute.side_effect = [invalid_json_response, invalid_json_response, invalid_json_response]

    result = await evaluator_subagent.evaluate_response("Test", "Response")

    assert result is None
    assert evaluator_subagent.sub_agent.execute.call_count == 3

@pytest.mark.asyncio
async def test_evaluate_response_with_policies(evaluator_subagent):
    evaluator_subagent.sub_agent.execute = AsyncMock()
    mock_json_response = '''{
      "usefulness": 8,
      "accuracy": 8,
      "completeness": 8,
      "emotional_adequacy": 8,
      "average_score": 8.0,
      "feedback": "Policy check passed"
    }'''
    evaluator_subagent.sub_agent.execute.return_value = mock_json_response

    policies = ["Be polite", "Be concise"]
    await evaluator_subagent.evaluate_response("Hello", "Hi!", policies=policies)

    # Verify that the task sent to sub_agent includes the policies
    call_kwargs = evaluator_subagent.sub_agent.execute.call_args.kwargs
    task_str = call_kwargs["task"]
    assert "Evaluate against these specific policies:" in task_str
    assert "- Be polite" in task_str
    assert "- Be concise" in task_str
