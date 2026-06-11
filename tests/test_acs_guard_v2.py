"""Tests for the ACS Runtime Safety Guard V2."""
import pytest
from unittest.mock import MagicMock
from magda_agent.safety.acs_guard_v2 import ACSGuardV2, SecurityViolationError
from magda_agent.safety.policy import PolicyLayer

def test_acs_guard_v2_valid_payload():
    guard = ACSGuardV2()
    payload = {
        "action": "allowed_action",
        "tool": "allowed_tool",
        "current_state": "idle",
        "next_state": "executing",
        "output": "safe output data"
    }
    result = guard.intercept_action(payload)
    assert result == payload

def test_acs_guard_v2_checkpoint_1_missing_action():
    guard = ACSGuardV2()
    payload = {"tool": "some_tool"}
    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 1: Input validation failed: missing 'action' field."):
        guard.intercept_action(payload)

def test_acs_guard_v2_checkpoint_1_empty_payload():
    guard = ACSGuardV2()
    payload = {}
    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 1: Input validation failed: workflow data is empty."):
        guard.intercept_action(payload)

def test_acs_guard_v2_checkpoint_2_unauthorized_intent():
    guard = ACSGuardV2()
    payload = {"action": "unauthorized_action"}
    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 2: Intent authorization failed: action 'unauthorized_action' is not allowed."):
        guard.intercept_action(payload)

def test_acs_guard_v2_checkpoint_3_forbidden_tool():
    guard = ACSGuardV2()
    payload = {"action": "some_action", "tool": "forbidden_tool"}
    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 3: Tool policy failed: tool 'forbidden_tool' is forbidden."):
        guard.intercept_action(payload)

def test_acs_guard_v2_checkpoint_3_policy_layer():
    mock_policy_layer = MagicMock()
    mock_policy_layer.evaluate.return_value = (False, "Mocked denial reason")

    guard = ACSGuardV2(policy_layer=mock_policy_layer)
    payload = {
        "action": "execute",
        "tool": "system_execute_code",
        "kwargs": {"code": "cat .env"}
    }
    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 3: Tool policy failed: Mocked denial reason"):
        guard.intercept_action(payload)

def test_acs_guard_v2_checkpoint_4_invalid_transition():
    guard = ACSGuardV2()
    payload = {
        "action": "some_action",
        "tool": "some_tool",
        "current_state": "error",
        "next_state": "executing"
    }
    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 4: State transition failed: cannot transition from 'error' to 'executing'."):
        guard.intercept_action(payload)

def test_acs_guard_v2_checkpoint_5_secret_leak():
    guard = ACSGuardV2()
    payload = {
        "action": "some_action",
        "tool": "some_tool",
        "current_state": "idle",
        "next_state": "executing",
        "output": "here is my secret_key"
    }
    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 5: Output sanitization failed: sensitive data detected in output."):
        guard.intercept_action(payload)
