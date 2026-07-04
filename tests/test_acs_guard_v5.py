import pytest
from unittest.mock import MagicMock
from magda_agent.safety.acs_guard_v5 import ACSGuardV5, SecurityViolationError


@pytest.fixture
def mock_policy_layer():
    policy = MagicMock()
    policy.evaluate.return_value = (True, "Allowed by mock")
    return policy


@pytest.fixture
def mock_audit_trail():
    return MagicMock()


@pytest.fixture
def acs_guard(mock_policy_layer, mock_audit_trail):
    return ACSGuardV5(policy_layer=mock_policy_layer, audit_trail=mock_audit_trail)


def test_checkpoint_1_input_validation(acs_guard: ACSGuardV5) -> None:
    """Tests the input validation checkpoint."""
    # Pass
    valid_data = {"action": "read", "tool": "ls"}
    passed, reason = acs_guard.checkpoint_1_input_validation(valid_data)
    assert passed
    assert "Passed" in reason

    # Fail - not a dict
    passed, reason = acs_guard.checkpoint_1_input_validation("not a dict")
    assert not passed
    assert "must be a dictionary" in reason

    # Fail - empty dict
    passed, reason = acs_guard.checkpoint_1_input_validation({})
    assert not passed
    assert "workflow data is empty" in reason

    # Fail - missing action
    passed, reason = acs_guard.checkpoint_1_input_validation({"tool": "ls"})
    assert not passed
    assert "missing 'action' field" in reason


def test_checkpoint_2_intent_authorization(acs_guard: ACSGuardV5) -> None:
    """Tests the intent authorization checkpoint."""
    # Pass
    assert acs_guard.checkpoint_2_intent_authorization({"action": "read"})[0]
    assert acs_guard.checkpoint_2_intent_authorization({"action": "write"})[0]
    assert acs_guard.checkpoint_2_intent_authorization({"action": "chat"})[0]

    # Fail - unauthorized_action
    passed, reason = acs_guard.checkpoint_2_intent_authorization({"action": "unauthorized_action"})
    assert not passed
    assert "explicitly blacklisted" in reason

    # Fail - not in allowed list
    passed, reason = acs_guard.checkpoint_2_intent_authorization({"action": "jump"})
    assert not passed
    assert "not in allowed intents list" in reason


def test_checkpoint_3_tool_policy(acs_guard: ACSGuardV5, mock_policy_layer: MagicMock) -> None:
    """Tests the tool policy checkpoint with mocking."""
    # Pass
    assert acs_guard.checkpoint_3_tool_policy({"tool": "ls"})[0]

    # Fail - forbidden_tool
    passed, reason = acs_guard.checkpoint_3_tool_policy({"tool": "forbidden_tool"})
    assert not passed
    assert "is forbidden" in reason

    # Fail - PolicyLayer denial
    mock_policy_layer.evaluate.return_value = (False, "Policy denial")
    passed, reason = acs_guard.checkpoint_3_tool_policy({"tool": "rm", "kwargs": {"path": "/"}})
    assert not passed
    assert "Policy denial" in reason


def test_checkpoint_4_state_transition(acs_guard: ACSGuardV5) -> None:
    """Tests the state transition checkpoint."""
    # Pass - default transition
    assert acs_guard.checkpoint_4_state_transition({"current_state": "idle", "next_state": "planning"})[0]

    # Pass - next_state not provided
    assert acs_guard.checkpoint_4_state_transition({"current_state": "idle"})[0]

    # Fail - unknown current_state
    passed, reason = acs_guard.checkpoint_4_state_transition({"current_state": "unknown", "next_state": "idle"})
    assert not passed
    assert "unknown current_state" in reason

    # Fail - invalid transition
    passed, reason = acs_guard.checkpoint_4_state_transition({"current_state": "idle", "next_state": "evaluating"})
    assert not passed
    assert "cannot transition" in reason

    # Pass - transition to error
    assert acs_guard.checkpoint_4_state_transition({"current_state": "planning", "next_state": "error"})[0]


def test_checkpoint_5_output_sanitization(acs_guard: ACSGuardV5) -> None:
    """Tests the output sanitization checkpoint."""
    # Pass
    assert acs_guard.checkpoint_5_output_sanitization({"output": "All good"})[0]
    assert acs_guard.checkpoint_5_output_sanitization({"output": None})[0]

    # Fail - sensitive data
    passed, reason = acs_guard.checkpoint_5_output_sanitization({"output": "Here is my secret_key: 12345"})
    assert not passed
    assert "sensitive pattern" in reason

    passed, reason = acs_guard.checkpoint_5_output_sanitization({"output": "Password is admin"})
    assert not passed
    assert "sensitive pattern" in reason

    # Regex patterns
    passed, reason = acs_guard.checkpoint_5_output_sanitization({"output": "Here is my .env file"})
    assert not passed
    assert "sensitive pattern" in reason

    passed, reason = acs_guard.checkpoint_5_output_sanitization({"output": "-----BEGIN RSA PRIVATE KEY-----"})
    assert not passed
    assert "sensitive pattern" in reason


def test_validate_action(acs_guard: ACSGuardV5) -> None:
    """Tests the comprehensive validation of all 5 checkpoints."""
    valid_data = {
        "action": "read",
        "tool": "ls",
        "current_state": "idle",
        "next_state": "planning",
        "output": "Some safe output"
    }
    assert acs_guard.validate_action(valid_data)

    invalid_data = valid_data.copy()
    invalid_data["action"] = "unauthorized_action"
    assert not acs_guard.validate_action(invalid_data)


def test_intercept_action(acs_guard: ACSGuardV5, mock_audit_trail: MagicMock) -> None:
    """Tests that actions are correctly intercepted and blocked if invalid."""
    valid_data = {
        "action": "read",
        "tool": "ls",
        "current_state": "idle",
        "next_state": "planning"
    }
    # Should not raise
    assert acs_guard.intercept_action(valid_data) == valid_data
    assert mock_audit_trail.log_call.called

    # Reset mock
    mock_audit_trail.log_call.reset_mock()

    invalid_data = valid_data.copy()
    invalid_data["tool"] = "forbidden_tool"

    with pytest.raises(SecurityViolationError) as excinfo:
        acs_guard.intercept_action(invalid_data)

    assert "ACS checkpoint 3" in str(excinfo.value)
    mock_audit_trail.log_call.assert_called_with(
        tool_name="forbidden_tool",
        kwargs={},
        why="ACS Checkpoint 3 Failed: Tool policy failed: tool 'forbidden_tool' is forbidden.",
        result="blocked",
        duration=0.0
    )
