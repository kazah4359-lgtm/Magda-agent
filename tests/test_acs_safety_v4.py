import pytest
from unittest.mock import patch, MagicMock
from magda_agent.safety.acs import ACSWorkflowGuard

@pytest.fixture
def acs_guard():
    return ACSWorkflowGuard()

def test_checkpoint_1_input_validation(acs_guard):
    assert acs_guard.checkpoint_1_input_validation({"action": "read"})[0] is True
    assert acs_guard.checkpoint_1_input_validation({})[0] is False
    assert acs_guard.checkpoint_1_input_validation({"tool": "something"})[0] is False

def test_checkpoint_2_intent_authorization(acs_guard):
    assert acs_guard.checkpoint_2_intent_authorization({"action": "read"})[0] is True
    assert acs_guard.checkpoint_2_intent_authorization({"action": "unauthorized_action"})[0] is False

def test_checkpoint_3_tool_policy(acs_guard):
    assert acs_guard.checkpoint_3_tool_policy({"tool": "ls"})[0] is True
    assert acs_guard.checkpoint_3_tool_policy({"tool": "forbidden_tool"})[0] is False

def test_checkpoint_4_state_transition(acs_guard):
    assert acs_guard.checkpoint_4_state_transition({"current_state": "idle", "next_state": "executing"})[0] is True
    assert acs_guard.checkpoint_4_state_transition({"current_state": "error", "next_state": "executing"})[0] is False

def test_checkpoint_5_output_sanitization(acs_guard):
    assert acs_guard.checkpoint_5_output_sanitization({"output": "hello world"})[0] is True
    assert acs_guard.checkpoint_5_output_sanitization({"output": "my secret_key is here"})[0] is False

@patch("magda_agent.safety.acs.logging")
def test_validate_workflow_success(mock_logging, acs_guard):
    valid_data = {
        "action": "read",
        "tool": "cat",
        "current_state": "idle",
        "next_state": "executing",
        "output": "safe content"
    }
    assert acs_guard.validate_workflow(valid_data) is True
    assert mock_logging.info.call_count == 5
    assert mock_logging.warning.call_count == 0

@patch("magda_agent.safety.acs.logging")
def test_validate_workflow_failure(mock_logging, acs_guard):
    invalid_data = {
        "action": "read",
        "tool": "forbidden_tool", # Fails at checkpoint 3
        "current_state": "idle",
        "next_state": "executing",
        "output": "safe content"
    }
    assert acs_guard.validate_workflow(invalid_data) is False
    assert mock_logging.info.call_count == 2 # Checkpoint 1 & 2 pass
    assert mock_logging.warning.call_count == 1 # Checkpoint 3 fails
