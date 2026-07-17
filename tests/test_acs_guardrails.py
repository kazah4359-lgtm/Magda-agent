import pytest
from unittest.mock import MagicMock
from magda_agent.safety.acs_guardrails import ACSGuardrailsV2, GuardrailViolationError
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

def test_validate_workflow_success(acs_guard):
    valid_data = {
        "action": "read",
        "tool": "cat",
        "current_state": "idle",
        "next_state": "planning",
        "output": "safe content"
    }
    assert acs_guard.validate_workflow(valid_data) is True

def test_validate_workflow_failure(acs_guard):
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


@pytest.fixture
def acs_guardrails_v2():
    return ACSGuardrailsV2()

def test_pre_tool_checkpoint_success(acs_guardrails_v2):
    passed, reason, feedback = acs_guardrails_v2.pre_tool_checkpoint("safe_tool", {"arg": "value"})
    assert passed is True
    assert reason == "Pre-tool checkpoint passed."
    assert feedback == {}

def test_pre_tool_checkpoint_invalid_name(acs_guardrails_v2):
    passed, reason, feedback = acs_guardrails_v2.pre_tool_checkpoint("", {"arg": "value"})
    assert passed is False
    assert "Input validation failed: workflow data is empty" in reason or "missing 'tool' field" in reason or "must be a string" in reason or "invalid tool name" in reason.lower()

    passed, reason, feedback = acs_guardrails_v2.pre_tool_checkpoint(None, {"arg": "value"})
    assert passed is False

def test_pre_tool_checkpoint_forbidden_tool(acs_guardrails_v2):
    passed, reason, feedback = acs_guardrails_v2.pre_tool_checkpoint("forbidden_tool", {"arg": "value"})
    assert passed is False
    assert "forbidden" in reason.lower()
    assert feedback["error"] == "Tool policy failed"

def test_pre_tool_checkpoint_malicious_arg(acs_guardrails_v2):
    passed, reason, feedback = acs_guardrails_v2.pre_tool_checkpoint("safe_tool", {"malicious_arg": "value"})
    assert passed is False
    assert "argument validation failed" in reason
    assert feedback["error"] == "Malicious argument detected"

def test_pre_output_checkpoint_success(acs_guardrails_v2):
    passed, reason, feedback = acs_guardrails_v2.pre_output_checkpoint("Safe output content.")
    assert passed is True
    assert reason == "Pre-output checkpoint passed."
    assert feedback == {}

def test_pre_output_checkpoint_none(acs_guardrails_v2):
    passed, reason, feedback = acs_guardrails_v2.pre_output_checkpoint(None)
    assert passed is True
    assert "Output sanitization passed" in reason or "Pre-output checkpoint passed" in reason

def test_pre_output_checkpoint_sensitive_data(acs_guardrails_v2):
    passed, reason, feedback = acs_guardrails_v2.pre_output_checkpoint("Here is your SECRET_TOKEN: 12345")
    assert passed is False
    assert "sensitive" in reason.lower() or "Output sanitization failed" in reason

    passed, reason, feedback = acs_guardrails_v2.pre_output_checkpoint("-----BEGIN RSA PRIVATE KEY-----")
    assert passed is False

def test_execute_with_guardrails_success(acs_guardrails_v2):
    mock_tool = MagicMock(return_value="Safe output")
    result = acs_guardrails_v2.execute_with_guardrails("safe_tool", {"arg": "value"}, mock_tool)

    assert result == "Safe output"
    mock_tool.assert_called_once_with(arg="value")

def test_execute_with_guardrails_pre_tool_fail(acs_guardrails_v2):
    mock_tool = MagicMock()
    with pytest.raises(GuardrailViolationError) as exc_info:
        acs_guardrails_v2.execute_with_guardrails("forbidden_tool", {"arg": "value"}, mock_tool)

    assert "forbidden" in str(exc_info.value).lower()
    mock_tool.assert_not_called()

def test_execute_with_guardrails_tool_exception(acs_guardrails_v2):
    mock_tool = MagicMock(side_effect=ValueError("Tool failed internally"))
    with pytest.raises(ValueError):
        acs_guardrails_v2.execute_with_guardrails("safe_tool", {"arg": "value"}, mock_tool)

    mock_tool.assert_called_once()

def test_execute_with_guardrails_pre_output_fail(acs_guardrails_v2):
    mock_tool = MagicMock(return_value="Here is your SECRET_TOKEN: 12345")
    with pytest.raises(GuardrailViolationError) as exc_info:
        acs_guardrails_v2.execute_with_guardrails("safe_tool", {"arg": "value"}, mock_tool)

    assert "sensitive" in str(exc_info.value).lower() or "Output sanitization failed" in str(exc_info.value)
    mock_tool.assert_called_once()
