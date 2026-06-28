import pytest
from unittest.mock import patch
from magda_agent.safety.acs_checkpoints_v3 import ACSCheckpointsV3

def test_acs_checkpoints_v3_pass():
    checkpoints = ACSCheckpointsV3()
    valid_data = {
        "action_name": "test_action",
        "tool_name": "allowed_tool",
        "state": "active",
        "output": "public result"
    }
    assert checkpoints.validate_action(valid_data) is True

def test_acs_checkpoints_v3_fail_1():
    checkpoints = ACSCheckpointsV3()
    assert checkpoints.validate_action({}) is False
    assert checkpoints.validate_action({"tool_name": "test"}) is False

    passed, reason = checkpoints.checkpoint_1_input_validation({})
    assert not passed
    assert "Checkpoint 1 Failed: empty action data." in reason

    passed, reason = checkpoints.checkpoint_1_input_validation({"tool_name": "test"})
    assert not passed
    assert "Checkpoint 1 Failed: missing 'action_name'." in reason

def test_acs_checkpoints_v3_fail_2():
    checkpoints = ACSCheckpointsV3()
    assert checkpoints.validate_action({"action_name": "unauthorized_action"}) is False

    passed, reason = checkpoints.checkpoint_2_intent_authorization({"action_name": "unauthorized_action"})
    assert not passed
    assert "Checkpoint 2 Failed: unauthorized action intent." in reason

def test_acs_checkpoints_v3_fail_3():
    checkpoints = ACSCheckpointsV3()
    assert checkpoints.validate_action({"action_name": "test", "tool_name": "forbidden_tool"}) is False

    passed, reason = checkpoints.checkpoint_3_tool_policy({"action_name": "test", "tool_name": "forbidden_tool"})
    assert not passed
    assert "Checkpoint 3 Failed: tool is forbidden." in reason

def test_acs_checkpoints_v3_fail_4():
    checkpoints = ACSCheckpointsV3()
    assert checkpoints.validate_action({"action_name": "test", "state": "error"}) is False

    passed, reason = checkpoints.checkpoint_4_state_transition({"action_name": "test", "state": "error"})
    assert not passed
    assert "Checkpoint 4 Failed: invalid state transition from error." in reason

def test_acs_checkpoints_v3_fail_5():
    checkpoints = ACSCheckpointsV3()
    assert checkpoints.validate_action({"action_name": "test", "output": "my secret_key is hidden"}) is False

    passed, reason = checkpoints.checkpoint_5_output_sanitization({"action_name": "test", "output": "my secret_key is hidden"})
    assert not passed
    assert "Checkpoint 5 Failed: sensitive data found in output." in reason

@patch('magda_agent.safety.acs_checkpoints_v3.logging.Logger.warning')
def test_acs_checkpoints_v3_logging(mock_warning):
    checkpoints = ACSCheckpointsV3()
    assert checkpoints.validate_action({}) is False
    mock_warning.assert_called_once_with("Checkpoint 1 Failed: empty action data.")
