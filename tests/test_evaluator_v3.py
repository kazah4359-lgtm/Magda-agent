import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.agents.evaluator_v3 import EvaluatorAgentV3

@pytest.mark.asyncio
async def test_evaluator_v3_agent_approve():
    """Tests the EvaluatorAgentV3 correctly parses a passing score from the LLM."""
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value='{"score": 9, "approved": true, "feedback": "Great response!"}')

    agent = EvaluatorAgentV3(llm=mock_llm)
    result = await agent.evaluate_generator_output("The sky is blue.", "What color is the sky?")

    assert result["score"] == 9
    assert result["approved"] is True
    assert result["feedback"] == "Great response!"
    mock_llm.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_evaluator_v3_agent_reject():
    """Tests the EvaluatorAgentV3 correctly parses a failing score from the LLM with markdown."""
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value='```json\n{"score": 4, "approved": false, "feedback": "Incorrect color."}\n```')

    agent = EvaluatorAgentV3(llm=mock_llm)
    result = await agent.evaluate_generator_output("The sky is green.", "What color is the sky?")

    assert result["score"] == 4
    assert result["approved"] is False
    assert result["feedback"] == "Incorrect color."
    mock_llm.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_evaluator_v3_agent_error_handling_invalid_json():
    """Tests the EvaluatorAgentV3 correctly handles JSON parsing errors."""
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value='This is not valid JSON')

    agent = EvaluatorAgentV3(llm=mock_llm)
    result = await agent.evaluate_generator_output("Some output", "Some request")

    assert result["score"] == 0
    assert result["approved"] is False
    assert "Evaluation error" in result["feedback"] or "Expecting value" in result["feedback"]
    mock_llm.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_evaluator_v3_agent_error_handling_missing_keys():
    """Tests the EvaluatorAgentV3 correctly handles JSON with missing keys."""
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value='{"score": 9, "approved": true}')

    agent = EvaluatorAgentV3(llm=mock_llm)
    result = await agent.evaluate_generator_output("Some output", "Some request")

    assert result["score"] == 0
    assert result["approved"] is False
    assert "Evaluation error: Missing required keys in LLM JSON output" in result["feedback"]
    mock_llm.chat_completion.assert_called_once()
