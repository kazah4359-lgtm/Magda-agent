import pytest
from magda_agent.safety.acs_checkpoints import ACSCheckpoints

def test_acs_checkpoints_pass():
    checkpoints = ACSCheckpoints()
    valid_data = {
        "action_name": "test_action",
        "tool_name": "allowed_tool",
        "state": "active",
        "output": "public result"
    }
    assert checkpoints.validate_action(valid_data) is True

def test_acs_checkpoints_fail_1():
    checkpoints = ACSCheckpoints()
    assert checkpoints.validate_action({}) is False
    assert checkpoints.validate_action({"tool_name": "test"}) is False

def test_acs_checkpoints_fail_2():
    checkpoints = ACSCheckpoints()
    assert checkpoints.validate_action({"action_name": "unauthorized_action"}) is False

def test_acs_checkpoints_fail_3():
    checkpoints = ACSCheckpoints()
    assert checkpoints.validate_action({"action_name": "test", "tool_name": "forbidden_tool"}) is False

def test_acs_checkpoints_fail_4():
    checkpoints = ACSCheckpoints()
    assert checkpoints.validate_action({"action_name": "test", "state": "error"}) is False

def test_acs_checkpoints_fail_5():
    checkpoints = ACSCheckpoints()
    assert checkpoints.validate_action({"action_name": "test", "output": "my secret_key is hidden"}) is False
