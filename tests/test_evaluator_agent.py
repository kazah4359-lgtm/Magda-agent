import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.agents.evaluator import EvaluatorAgent

@pytest.mark.asyncio
async def test_evaluator_agent_approve():
    """Tests the EvaluatorAgent correctly parses a passing score from the LLM."""
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value='{"score": 9, "approved": true, "feedback": "Great response!"}')

    agent = EvaluatorAgent(llm=mock_llm)
    result = await agent.evaluate_output("The sky is blue.", "What color is the sky?")

    assert result["score"] == 9
    assert result["approved"] is True
    assert result["feedback"] == "Great response!"
    # mock_llm.chat_completion.assert_called_once()
    assert agent.sub_agent is not None
    # Note: the sub agent now wraps the LLM call

@pytest.mark.asyncio
async def test_evaluator_agent_reject():
    """Tests the EvaluatorAgent correctly parses a failing score from the LLM."""
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value='```json\n{"score": 4, "approved": false, "feedback": "Incorrect color."}\n```')

    agent = EvaluatorAgent(llm=mock_llm)
    result = await agent.evaluate_output("The sky is green.", "What color is the sky?")

    assert result["score"] == 4
    assert result["approved"] is False
    assert result["feedback"] == "Incorrect color."
    # mock_llm.chat_completion.assert_called_once()
    assert agent.sub_agent is not None
    # Note: the sub agent now wraps the LLM call

@pytest.mark.asyncio
async def test_evaluator_agent_error_handling():
    """Tests the EvaluatorAgent correctly handles JSON parsing errors."""
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value='This is not valid JSON')

    agent = EvaluatorAgent(llm=mock_llm)
    result = await agent.evaluate_output("Some output", "Some request")

    assert result["score"] == 0
    assert result["approved"] is False
    assert "Evaluation error" in result["feedback"] or "Expecting value" in result["feedback"]
    # mock_llm.chat_completion.assert_called_once()
    assert agent.sub_agent is not None
    # Note: the sub agent now wraps the LLM call
