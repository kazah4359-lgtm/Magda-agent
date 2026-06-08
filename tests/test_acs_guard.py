"""Tests for the ACS Runtime Safety Guard."""
import pytest
from magda_agent.safety.acs_guard import ACSGuard, SecurityViolationError

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
