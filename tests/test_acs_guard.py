"""Tests for the ACS Runtime Safety Guard."""
import pytest
from magda_agent.safety.acs_guard import (
    ACSGuard,
    SecurityViolationError,
    InputValidationCheckpoint,
    IntentAuthorizationCheckpoint,
    ToolPolicyCheckpoint,
    StateTransitionCheckpoint,
    OutputSanitizationCheckpoint
)

def test_acs_guard_valid_payload():
    guard = ACSGuard()
    payload = {
        "action": "allowed_action",
        "tool": "allowed_tool",
        "current_state": "idle",
        "next_state": "executing",
        "output": "safe output data"
    }
    result = guard.intercept_action(payload)
    assert result == payload

def test_acs_guard_checkpoint_1_missing_action():
    guard = ACSGuard()
    payload = {"tool": "some_tool"}
    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 1: Input validation failed: missing 'action' field."):
        guard.intercept_action(payload)

def test_acs_guard_checkpoint_1_empty_payload():
    guard = ACSGuard()
    payload = {}
    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 1: Input validation failed: workflow data is empty."):
        guard.intercept_action(payload)

def test_acs_guard_checkpoint_2_unauthorized_intent():
    guard = ACSGuard()
    payload = {"action": "unauthorized_action"}
    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 2: Intent authorization failed: action 'unauthorized_action' is not allowed."):
        guard.intercept_action(payload)

def test_acs_guard_checkpoint_3_forbidden_tool():
    guard = ACSGuard()
    payload = {"action": "some_action", "tool": "forbidden_tool"}
    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 3: Tool policy failed: tool 'forbidden_tool' is forbidden."):
        guard.intercept_action(payload)

def test_acs_guard_checkpoint_4_invalid_transition():
    guard = ACSGuard()
    payload = {
        "action": "some_action",
        "tool": "some_tool",
        "current_state": "error",
        "next_state": "executing"
    }
    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 4: State transition failed: cannot transition from 'error' to 'executing'."):
        guard.intercept_action(payload)

def test_acs_guard_checkpoint_5_secret_leak():
    guard = ACSGuard()
    payload = {
        "action": "some_action",
        "tool": "some_tool",
        "current_state": "idle",
        "next_state": "executing",
        "output": "here is my secret_key"
    }
    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 5: Output sanitization failed: sensitive data detected in output."):
        guard.intercept_action(payload)

# Individual Checkpoint Tests
def test_input_validation_checkpoint():
    checkpoint = InputValidationCheckpoint()
    passed, reason = checkpoint.validate({"action": "run"})
    assert passed
    assert reason == "Input validation passed."

    passed, reason = checkpoint.validate({})
    assert not passed
    assert reason == "Input validation failed: workflow data is empty."

    passed, reason = checkpoint.validate({"tool": "ls"})
    assert not passed
    assert reason == "Input validation failed: missing 'action' field."

def test_intent_authorization_checkpoint():
    checkpoint = IntentAuthorizationCheckpoint()
    passed, reason = checkpoint.validate({"action": "run"})
    assert passed
    assert reason == "Intent authorization passed."

    passed, reason = checkpoint.validate({"action": "unauthorized_action"})
    assert not passed
    assert reason == "Intent authorization failed: action 'unauthorized_action' is not allowed."

def test_tool_policy_checkpoint():
    checkpoint = ToolPolicyCheckpoint()
    passed, reason = checkpoint.validate({"tool": "ls"})
    assert passed
    assert reason == "Tool policy passed."

    passed, reason = checkpoint.validate({"tool": "forbidden_tool"})
    assert not passed
    assert reason == "Tool policy failed: tool 'forbidden_tool' is forbidden."

def test_state_transition_checkpoint():
    checkpoint = StateTransitionCheckpoint()
    passed, reason = checkpoint.validate({"current_state": "idle", "next_state": "executing"})
    assert passed
    assert reason == "State transition passed."

    passed, reason = checkpoint.validate({"current_state": "error", "next_state": "executing"})
    assert not passed
    assert reason == "State transition failed: cannot transition from 'error' to 'executing'."

def test_output_sanitization_checkpoint():
    checkpoint = OutputSanitizationCheckpoint()
    passed, reason = checkpoint.validate({"output": "safe output"})
    assert passed
    assert reason == "Output sanitization passed."

    passed, reason = checkpoint.validate({"output": "This is a secret_key value"})
    assert not passed
    assert reason == "Output sanitization failed: sensitive data detected in output."
