import pytest
from unittest.mock import MagicMock, patch

from magda_agent.safety.acs_fallback_v5 import (
    ACSControlFallbackV5,
    ACSFallbackV5,
    DEFAULT_NEUTRAL_RESPONSE,
)
from magda_agent.safety.guardrails import SecurityViolationError


def test_acs_fallback_v5_init():
    fallback = ACSControlFallbackV5()
    assert fallback.neutral_response == DEFAULT_NEUTRAL_RESPONSE


def test_validate_state_and_output_passed():
    fallback = ACSControlFallbackV5()
    action_data = {
        "current_state": "idle",
        "next_state": "planning",
        "action_name": "read",
        "tool_name": "read_file",
    }
    output = "Safe text content"
    passed, reason = fallback.validate_state_and_output(action_data, output)
    assert passed is True
    assert "passed" in reason.lower()


def test_validate_state_and_output_denied_state():
    fallback = ACSControlFallbackV5()
    action_data = {
        "current_state": "unknown_state",
        "next_state": "executing",
        "action_name": "read",
        "tool_name": "read_file",
    }
    output = "Safe text content"
    passed, reason = fallback.validate_state_and_output(action_data, output)
    assert passed is False
    assert "State Check Denied" in reason


def test_validate_state_and_output_denied_output():
    fallback = ACSControlFallbackV5()
    action_data = {
        "current_state": "idle",
        "next_state": "planning",
        "action_name": "read",
        "tool_name": "read_file",
    }
    # Sensitive pattern trigger: SSN pattern or credit card
    output = "User SSN is 123-45-6789"
    passed, reason = fallback.validate_state_and_output(action_data, output)
    assert passed is False
    assert "Output Validation Denied" in reason


def test_execute_with_fallback_success_sync():
    fallback = ACSControlFallbackV5()

    def dummy_tool(text: str):
        return f"Hello {text}"

    action_data = {
        "current_state": "idle",
        "next_state": "planning",
        "kwargs": {"text": "world"},
    }

    success, result = fallback.execute_with_fallback(dummy_tool, action_data)
    assert success is True
    assert result == "Hello world"


def test_execute_with_fallback_denied_output_returns_neutral():
    fallback = ACSControlFallbackV5()

    def dummy_tool():
        return "Confidential info: SSN 000-11-2222"

    action_data = {
        "current_state": "idle",
        "next_state": "planning",
        "kwargs": {},
    }

    success, result = fallback.execute_with_fallback(dummy_tool, action_data)
    assert success is False
    assert result == DEFAULT_NEUTRAL_RESPONSE


def test_execute_with_fallback_denied_state_returns_neutral():
    fallback = ACSControlFallbackV5()

    def dummy_tool():
        return "Normal response"

    action_data = {
        "current_state": "invalid_state",
        "next_state": "executing",
        "kwargs": {},
    }

    success, result = fallback.execute_with_fallback(dummy_tool, action_data)
    assert success is False
    assert result == DEFAULT_NEUTRAL_RESPONSE


def test_execute_with_fallback_custom_neutral_dict():
    custom_resp = {"status": "blocked", "message": "Neutral response fallback"}
    fallback = ACSControlFallbackV5(neutral_response=custom_resp)

    def dummy_tool():
        return "Sensitive SSN 999-88-7777"

    action_data = {
        "current_state": "idle",
        "next_state": "planning",
        "kwargs": {},
    }

    success, result = fallback.execute_with_fallback(dummy_tool, action_data)
    assert success is False
    assert isinstance(result, dict)
    assert result["status"] == "blocked"
    assert result["message"] == "Neutral response fallback"


def test_execute_with_fallback_security_violation_exception():
    fallback = ACSControlFallbackV5()

    def dummy_tool():
        raise SecurityViolationError("Explicit policy violation in tool execution")

    action_data = {
        "current_state": "idle",
        "next_state": "planning",
        "kwargs": {},
    }

    success, result = fallback.execute_with_fallback(dummy_tool, action_data)
    assert success is False
    assert result == DEFAULT_NEUTRAL_RESPONSE


@pytest.mark.asyncio
async def test_execute_with_fallback_async_success():
    fallback = ACSControlFallbackV5()

    async def async_dummy_tool(val: int):
        return f"Result: {val * 2}"

    action_data = {
        "current_state": "idle",
        "next_state": "planning",
        "kwargs": {"val": 21},
    }

    success, result = await fallback.execute_with_fallback_async(async_dummy_tool, action_data)
    assert success is True
    assert result == "Result: 42"


@pytest.mark.asyncio
async def test_execute_with_fallback_async_denied():
    fallback = ACSControlFallbackV5()

    async def async_dummy_tool():
        return "User secret SSN 123-45-6789"

    action_data = {
        "current_state": "idle",
        "next_state": "planning",
        "kwargs": {},
    }

    success, result = await fallback.execute_with_fallback_async(async_dummy_tool, action_data)
    assert success is False
    assert result == DEFAULT_NEUTRAL_RESPONSE
