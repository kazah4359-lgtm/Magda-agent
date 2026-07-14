import pytest
from magda_agent.safety.acs_checkpoints import ACSCheckpoints, SecurityViolationError, CheckpointStage, Checkpoint
from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.audit_trail import AuditTrail

def test_acs_checkpoints_pass():
    checkpoints = ACSCheckpoints()
    valid_data = {
        "action_name": "test_action",
        "tool_name": "allowed_tool",
        "state": "active",
        "output": "public result"
    }
    assert checkpoints.validate_action(valid_data) is True

def test_acs_checkpoints_fail_1_input():
    checkpoints = ACSCheckpoints()
    assert checkpoints.validate_action({}) is False
    assert checkpoints.validate_action({"tool_name": "test"}) is False
    assert checkpoints.validate_action({"action_name": 123, "tool_name": "test"}) is False

def test_acs_checkpoints_fail_2_intent():
    checkpoints = ACSCheckpoints()
    assert checkpoints.validate_action({"action_name": "unauthorized_action", "tool_name": "test"}) is False
    assert checkpoints.validate_action({"action_name": "hack", "tool_name": "test"}) is False

def test_acs_checkpoints_fail_3_tool_policy():
    policy = PolicyLayer()
    checkpoints = ACSCheckpoints(policy_layer=policy)
    # forbidden_tool is hardcoded in ACSCheckpoints
    assert checkpoints.validate_action({"action_name": "execute", "tool_name": "forbidden_tool"}) is False

    # Tool denied by PolicyLayer (e.g. programmer with sensitive code)
    assert checkpoints.validate_action({
        "action_name": "execute",
        "tool_name": "programmer",
        "kwargs": {"code": "read .env file"}
    }) is False

def test_acs_checkpoints_fail_4_state():
    checkpoints = ACSCheckpoints()
    # Invalid transition
    assert checkpoints.validate_action({
        "action_name": "test_action",
        "tool_name": "test",
        "state": "error",
        "next_state": "executing"
    }) is False
    # Unknown state
    assert checkpoints.validate_action({
        "action_name": "test_action",
        "tool_name": "test",
        "state": "unknown"
    }) is False

def test_acs_checkpoints_fail_5_output():
    checkpoints = ACSCheckpoints()
    assert checkpoints.validate_action({
        "action_name": "test_action",
        "tool_name": "test",
        "output": "my secret_key is hidden"
    }) is False
    assert checkpoints.validate_action({
        "action_name": "test_action",
        "tool_name": "test",
        "output": "FOUND API_KEY=12345"
    }) is False

def test_intercept_action():
    checkpoints = ACSCheckpoints()
    valid_data = {
        "action_name": "test_action",
        "tool_name": "allowed_tool"
    }
    assert checkpoints.intercept_action(valid_data) == valid_data

    with pytest.raises(SecurityViolationError):
        checkpoints.intercept_action({"action_name": "hack", "tool_name": "test"})

def test_audit_trail_logging():
    audit = AuditTrail(db_path=None)
    checkpoints = ACSCheckpoints(audit_trail=audit)

    checkpoints.validate_action({"action_name": "hack", "tool_name": "test"})
    assert len(audit.get_all()) == 1
    assert audit.get_all()[0]["result"] == "blocked"

    checkpoints.validate_action({"action_name": "chat", "tool_name": "test"})
    # It adds two logs: one for the passing of pre/post execution if we call it via validate_action
    # Wait, validate_action calls validate_pre_execution and validate_post_execution.
    # validate_pre_execution logs ONLY on failure.
    # validate_post_execution logs ONLY on failure.
    # validate_action logs on SUCCESS.

    assert len(audit.get_all()) == 2
    assert audit.get_all()[1]["result"] == "allowed"

def test_checkpoint_stage_enum():
    assert CheckpointStage.INPUT.value == "input"
    assert CheckpointStage.EXECUTION.value == "execution"
    assert CheckpointStage.OUTPUT.value == "output"

def test_checkpoint_class():
    def dummy_validator(action_data):
        return True, "Passed"

    cp = Checkpoint(name="Dummy", stage=CheckpointStage.INPUT, validate_func=dummy_validator)
    assert cp.name == "Dummy"
    assert cp.stage == CheckpointStage.INPUT
    assert cp.run({}) == (True, "Passed")

def test_run_stage_explicitly():
    checkpoints = ACSCheckpoints()
    valid_data = {
        "action_name": "test_action",
        "tool_name": "allowed_tool"
    }
    # INPUT stage passes
    passed, msg = checkpoints._run_stage(CheckpointStage.INPUT, valid_data)
    assert passed is True

    # Invalid input
    invalid_data = {}
    passed, msg = checkpoints._run_stage(CheckpointStage.INPUT, invalid_data)
    assert passed is False
    assert "Checkpoint 1 Failed" in msg
