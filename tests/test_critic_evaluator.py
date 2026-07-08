import pytest
import json
from unittest.mock import AsyncMock

from magda_agent.agents.critic_evaluator import CriticEvaluator
from magda_agent.llm_client import LLMClient


@pytest.mark.asyncio
async def test_critic_evaluator_success() -> None:
    """
    Test that CriticEvaluator correctly parses valid JSON feedback and returns expected scores.
    """
    mock_llm = AsyncMock(spec=LLMClient)
    expected_result = {
        "usefulness": 8,
        "accuracy": 9,
        "completeness": 7,
        "emotional_adequacy": 8,
        "average_score": 8.0,
        "feedback": "Good response"
    }

    mock_llm.chat_completion.return_value = json.dumps(expected_result)

    evaluator = CriticEvaluator(llm=mock_llm)

    result = await evaluator.evaluate_generator_output("user input", "generator output")

    assert result == expected_result
    assert evaluator.last_evaluation == expected_result
    mock_llm.chat_completion.assert_called_once()


@pytest.mark.asyncio
async def test_critic_evaluator_json_parsing_failure() -> None:
    """
    Test that CriticEvaluator handles JSON parsing failures by retrying and eventually returning None.
    """
    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.chat_completion.return_value = "invalid json format"

    evaluator = CriticEvaluator(llm=mock_llm)

    result = await evaluator.evaluate_generator_output("user input", "generator output")

    assert result is None
    assert mock_llm.chat_completion.call_count == 3


@pytest.mark.asyncio
async def test_critic_evaluator_missing_keys_failure() -> None:
    """
    Test that CriticEvaluator handles missing keys in JSON output.
    """
    mock_llm = AsyncMock(spec=LLMClient)
    invalid_result = {
        "usefulness": 8,
        "accuracy": 9,
        # missing keys
    }
    mock_llm.chat_completion.return_value = json.dumps(invalid_result)

    evaluator = CriticEvaluator(llm=mock_llm)

    result = await evaluator.evaluate_generator_output("user input", "generator output")

    assert result is None
    assert mock_llm.chat_completion.call_count == 3


@pytest.mark.asyncio
async def test_critic_evaluator_markdown_json() -> None:
    """
    Test that CriticEvaluator correctly strips markdown formatting from JSON output.
    """
    mock_llm = AsyncMock(spec=LLMClient)
    expected_result = {
        "usefulness": 8,
        "accuracy": 9,
        "completeness": 7,
        "emotional_adequacy": 8,
        "average_score": 8.0,
        "feedback": "Good response"
    }

    mock_llm.chat_completion.return_value = f"```json\n{json.dumps(expected_result)}\n```"

    evaluator = CriticEvaluator(llm=mock_llm)

    result = await evaluator.evaluate_generator_output("user input", "generator output")

    assert result == expected_result
    mock_llm.chat_completion.assert_called_once()
