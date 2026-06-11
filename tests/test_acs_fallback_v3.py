import pytest
from unittest.mock import patch
from magda_agent.safety.acs import ACSWorkflowGuard

@pytest.fixture
def acs_guard():
    return ACSWorkflowGuard()

@patch("magda_agent.safety.acs.logging")
def test_validate_with_fallback_success(mock_logging, acs_guard):
    valid_data = {
        "action": "read",
        "tool": "cat",
        "current_state": "idle",
        "next_state": "executing",
        "output": "safe content"
    }
    fallback_action = {"action": "fallback"}
    passed, result = acs_guard.validate_with_fallback(valid_data, fallback_action)
    assert passed is True
    assert result == valid_data

@patch("magda_agent.safety.acs.logging")
def test_validate_with_fallback_failure_with_fallback(mock_logging, acs_guard):
    invalid_data = {
        "action": "read",
        "tool": "forbidden_tool",
        "current_state": "idle",
        "next_state": "executing",
        "output": "safe content"
    }
    fallback_action = {"action": "safe_fallback_tool"}
    passed, result = acs_guard.validate_with_fallback(invalid_data, fallback_action)
    assert passed is False
    assert result == fallback_action
    mock_logging.info.assert_any_call("ACS validation failed, triggering fallback action.")

@patch("magda_agent.safety.acs.logging")
def test_validate_with_fallback_failure_without_fallback(mock_logging, acs_guard):
    invalid_data = {
        "action": "read",
        "tool": "forbidden_tool",
        "current_state": "idle",
        "next_state": "executing",
        "output": "safe content"
    }
    passed, result = acs_guard.validate_with_fallback(invalid_data)
    assert passed is False
    assert result["action"] == "error"
    assert result["next_state"] == "error"
    mock_logging.warning.assert_any_call("ACS validation failed and no fallback provided. Returning error state.")
