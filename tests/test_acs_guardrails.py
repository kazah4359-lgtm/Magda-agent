import pytest
from unittest.mock import patch, MagicMock
from magda_agent.safety.acs import ACSWorkflowGuard, SecurityViolationError

@pytest.fixture
def acs_guard():
    return ACSWorkflowGuard()

def test_checkpoint_1_input_validation(acs_guard):
    assert acs_guard.checkpoint_1_input_validation({"action": "read", "tool": "cat"})[0] is True
    assert acs_guard.checkpoint_1_input_validation({})[0] is False
    assert acs_guard.checkpoint_1_input_validation({"tool": "cat"})[0] is False
    assert acs_guard.checkpoint_1_input_validation({"action": 123, "tool": "cat"})[0] is False
    assert acs_guard.checkpoint_1_input_validation("not a dict")[0] is False

def test_checkpoint_2_intent_authorization(acs_guard):
    assert acs_guard.checkpoint_2_intent_authorization({"action": "read"})[0] is True
    assert acs_guard.checkpoint_2_intent_authorization({"action": "write"})[0] is True
    assert acs_guard.checkpoint_2_intent_authorization({"action": "unauthorized_action"})[0] is False
    assert acs_guard.checkpoint_2_intent_authorization({"action": "unknown_intent"})[0] is False

def test_checkpoint_3_tool_policy(acs_guard):
    assert acs_guard.checkpoint_3_tool_policy({"tool": "ls"})[0] is True
    assert acs_guard.checkpoint_3_tool_policy({"tool": "forbidden_tool"})[0] is False

    with patch.object(acs_guard.policy_layer, 'evaluate', return_value=(False, "Denied by policy")):
        assert acs_guard.checkpoint_3_tool_policy({"tool": "some_tool"})[0] is False

def test_checkpoint_4_state_transition(acs_guard):
    assert acs_guard.checkpoint_4_state_transition({"current_state": "idle", "next_state": "planning"})[0] is True
    assert acs_guard.checkpoint_4_state_transition({"current_state": "planning", "next_state": "executing"})[0] is True
    assert acs_guard.checkpoint_4_state_transition({"current_state": "executing", "next_state": "evaluating"})[0] is True
    assert acs_guard.checkpoint_4_state_transition({"current_state": "evaluating", "next_state": "idle"})[0] is True

    assert acs_guard.checkpoint_4_state_transition({"current_state": "idle", "next_state": "executing"})[0] is False
    assert acs_guard.checkpoint_4_state_transition({"current_state": "error", "next_state": "idle"})[0] is True
    assert acs_guard.checkpoint_4_state_transition({"current_state": "idle", "next_state": "unknown"})[0] is False

def test_checkpoint_5_output_sanitization(acs_guard):
    assert acs_guard.checkpoint_5_output_sanitization({"output": "hello world"})[0] is True
    assert acs_guard.checkpoint_5_output_sanitization({"output": "my secret_key is here"})[0] is False
    assert acs_guard.checkpoint_5_output_sanitization({"output": "my api_key: 12345"})[0] is False
    assert acs_guard.checkpoint_5_output_sanitization({"output": "-----BEGIN RSA PRIVATE KEY-----"})[0] is False

@patch("magda_agent.safety.acs.logging")
def test_validate_workflow_success(mock_logging, acs_guard):
    valid_data = {
        "action": "read",
        "tool": "cat",
        "current_state": "idle",
        "next_state": "planning",
        "output": "safe content"
    }
    assert acs_guard.validate_workflow(valid_data) is True

@patch("magda_agent.safety.acs.logging")
def test_validate_workflow_failure(mock_logging, acs_guard):
    invalid_data = {
        "action": "read",
        "tool": "forbidden_tool",
        "current_state": "idle",
        "next_state": "planning",
        "output": "safe content"
    }
    assert acs_guard.validate_workflow(invalid_data) is False

def test_intercept_action_success(acs_guard):
    valid_data = {
        "action": "read",
        "tool": "cat",
        "current_state": "idle",
        "next_state": "planning",
        "output": "safe content"
    }
    result = acs_guard.intercept_action(valid_data)
    assert result == valid_data

def test_intercept_action_failure(acs_guard):
    invalid_data = {
        "action": "read",
        "tool": "forbidden_tool",
        "current_state": "idle",
        "next_state": "planning",
        "output": "safe content"
    }
    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 3"):
        acs_guard.intercept_action(invalid_data)
