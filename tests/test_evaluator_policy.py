import pytest
from unittest.mock import AsyncMock
from typing import Dict, Any

from magda_agent.agents.evaluator_policy import PolicyEnforcedEvaluator

class MockEvaluator:
    """Mock underlying evaluator."""
    def __init__(self, expected_response: Dict[str, Any]):
        self.evaluate_generator_output = AsyncMock(return_value=expected_response)


@pytest.mark.asyncio
async def test_policy_enforced_evaluator_passes_policies():
    """Test that output passing policies is delegated to the LLM evaluator."""
    expected_response = {"score": 9, "approved": True, "feedback": "Great output."}
    mock_evaluator = MockEvaluator(expected_response=expected_response)

    policy_evaluator = PolicyEnforcedEvaluator(evaluator_agent=mock_evaluator)

    output = "This is a clean and compliant output that contains the mandatory_term."
    policies = ["must_contain:mandatory_term", "max_length:200"]

    result = await policy_evaluator.evaluate_generator_output(
        generator_output=output,
        user_request="Write a clean output.",
        policies=policies
    )

    assert result == expected_response
    mock_evaluator.evaluate_generator_output.assert_called_once_with(output, "Write a clean output.")

@pytest.mark.asyncio
async def test_policy_enforced_evaluator_fails_forbidden():
    """Test that output containing forbidden terms is rejected without calling the LLM evaluator."""
    mock_evaluator = MockEvaluator(expected_response={"score": 10, "approved": True, "feedback": "Should not be called."})

    policy_evaluator = PolicyEnforcedEvaluator(evaluator_agent=mock_evaluator)

    output = "This output contains a malicious_script."
    policies = ["forbidden:malicious_script"]

    result = await policy_evaluator.evaluate_generator_output(
        generator_output=output,
        user_request="Write an output.",
        policies=policies
    )

    assert result["score"] == 0
    assert result["approved"] is False
    assert "Policy Violation" in result["feedback"]
    assert "forbidden term" in result["feedback"]

    mock_evaluator.evaluate_generator_output.assert_not_called()

@pytest.mark.asyncio
async def test_policy_enforced_evaluator_fails_missing_term():
    """Test that output missing required terms is rejected."""
    mock_evaluator = MockEvaluator(expected_response={"score": 10, "approved": True, "feedback": "Should not be called."})

    policy_evaluator = PolicyEnforcedEvaluator(evaluator_agent=mock_evaluator)

    output = "This output missed the required term."
    policies = ["must_contain:mandatory_term"]

    result = await policy_evaluator.evaluate_generator_output(
        generator_output=output,
        user_request="Write an output.",
        policies=policies
    )

    assert result["score"] == 0
    assert result["approved"] is False
    assert "Missing required term" in result["feedback"]

    mock_evaluator.evaluate_generator_output.assert_not_called()

@pytest.mark.asyncio
async def test_policy_enforced_evaluator_fails_max_length():
    """Test that output exceeding max length is rejected."""
    mock_evaluator = MockEvaluator(expected_response={"score": 10, "approved": True, "feedback": "Should not be called."})

    policy_evaluator = PolicyEnforcedEvaluator(evaluator_agent=mock_evaluator)

    output = "This output is way too long. " * 10
    policies = ["max_length:20"]

    result = await policy_evaluator.evaluate_generator_output(
        generator_output=output,
        user_request="Write a short output.",
        policies=policies
    )

    assert result["score"] == 0
    assert result["approved"] is False
    assert "Exceeds maximum length" in result["feedback"]

    mock_evaluator.evaluate_generator_output.assert_not_called()
