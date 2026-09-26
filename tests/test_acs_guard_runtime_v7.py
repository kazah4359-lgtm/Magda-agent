import pytest
from unittest.mock import MagicMock
from magda_agent.safety.acs_guard_runtime_v7 import (
    ACSGuardRuntimeV7,
    ACSGuard,
    ACSGuardV7,
    SecurityViolationError,
    ACSRuntimeViolationError
)
from magda_agent.safety.taint import mark_tainted


@pytest.fixture
def mock_policy_layer():
    policy = MagicMock()
    policy.evaluate.return_value = (True, "Allowed by mock policy")
    return policy


@pytest.fixture
def mock_audit_trail():
    return MagicMock()


@pytest.fixture
def mock_persistence():
    return MagicMock()


@pytest.fixture
def acs_guard(mock_policy_layer, mock_audit_trail, mock_persistence):
    return ACSGuardRuntimeV7(
        policy_layer=mock_policy_layer,
        audit_trail=mock_audit_trail,
        persistence=mock_persistence
    )


def test_aliases():
    """Verify class aliases match the primary class."""
    assert ACSGuard is ACSGuardRuntimeV7
    assert ACSGuardV7 is ACSGuardRuntimeV7


def test_checkpoint_1_input_validation(acs_guard: ACSGuardRuntimeV7) -> None:
    """Tests Checkpoint 1: Input Validation."""
    valid_data = {"action": "read", "tool": "ls"}
    passed, reason = acs_guard.checkpoint_1_input_validation(valid_data)
    assert passed
    assert "passed" in reason.lower()

    # None kwargs handled gracefully
    none_kwargs_data = {"action": "read", "tool": "ls", "kwargs": None}
    passed, reason = acs_guard.checkpoint_1_input_validation(none_kwargs_data)
    assert passed

    # Non-dictionary input
    passed, reason = acs_guard.checkpoint_1_input_validation("not a dict")
    assert not passed
    assert "must be a dictionary" in reason

    # Empty dictionary
    passed, reason = acs_guard.checkpoint_1_input_validation({})
    assert not passed
    assert "empty" in reason

    # Missing action field
    passed, reason = acs_guard.checkpoint_1_input_validation({"tool": "ls"})
    assert not passed
    assert "missing 'action' field" in reason

    # Non-string tool field
    passed, reason = acs_guard.checkpoint_1_input_validation({"action": "read", "tool": 123})
    assert not passed
    assert "'tool' must be a string" in reason

    # Tainted kwargs
    tainted_kwargs = mark_tainted({"path": "/etc/shadow"})
    passed, reason = acs_guard.checkpoint_1_input_validation({
        "action": "read",
        "tool": "cat",
        "kwargs": tainted_kwargs
    })
    assert not passed
    assert "tainted data detected" in reason


def test_checkpoint_2_intent_authorization(acs_guard: ACSGuardRuntimeV7) -> None:
    """Tests Checkpoint 2: Intent Authorization."""
    for allowed_action in ["read", "write", "execute", "plan", "reflect", "delegate", "analyze", "chat", "search"]:
        passed, _ = acs_guard.checkpoint_2_intent_authorization({"action": allowed_action})
        assert passed

    # Blacklisted action
    passed, reason = acs_guard.checkpoint_2_intent_authorization({"action": "unauthorized_action"})
    assert not passed
    assert "blacklisted" in reason

    # Unknown action
    passed, reason = acs_guard.checkpoint_2_intent_authorization({"action": "jump_around"})
    assert not passed
    assert "not in allowed intents list" in reason

    # Tainted action
    passed, reason = acs_guard.checkpoint_2_intent_authorization({"action": mark_tainted("read")})
    assert not passed
    assert "action is tainted" in reason


def test_checkpoint_3_tool_policy(acs_guard: ACSGuardRuntimeV7, mock_policy_layer: MagicMock) -> None:
    """Tests Checkpoint 3: Tool Policy."""
    passed, _ = acs_guard.checkpoint_3_tool_policy({"tool": "ls"})
    assert passed

    # None kwargs handled safely without TypeError
    passed, _ = acs_guard.checkpoint_3_tool_policy({"tool": "ls", "kwargs": None})
    assert passed

    # Tainted tool name
    passed, reason = acs_guard.checkpoint_3_tool_policy({"tool": mark_tainted("ls")})
    assert not passed
    assert "tool name is tainted" in reason

    # Forbidden tool
    passed, reason = acs_guard.checkpoint_3_tool_policy({"tool": "forbidden_tool"})
    assert not passed
    assert "is forbidden" in reason

    # PolicyLayer denial
    mock_policy_layer.evaluate.return_value = (False, "Policy denied execution")
    passed, reason = acs_guard.checkpoint_3_tool_policy({"tool": "rm", "kwargs": {"path": "/"}})
    assert not passed
    assert "Policy denied execution" in reason


def test_checkpoint_4_state_transition(acs_guard: ACSGuardRuntimeV7) -> None:
    """Tests Checkpoint 4: State Transition."""
    passed, _ = acs_guard.checkpoint_4_state_transition({"current_state": "idle", "next_state": "planning"})
    assert passed

    passed, _ = acs_guard.checkpoint_4_state_transition({"current_state": "idle"})
    assert passed

    # Unknown current state
    passed, reason = acs_guard.checkpoint_4_state_transition({"current_state": "invalid_state", "next_state": "idle"})
    assert not passed
    assert "unknown current_state" in reason

    # Invalid state transition
    passed, reason = acs_guard.checkpoint_4_state_transition({"current_state": "idle", "next_state": "evaluating"})
    assert not passed
    assert "cannot transition" in reason


def test_checkpoint_5_output_sanitization(acs_guard: ACSGuardRuntimeV7) -> None:
    """Tests Checkpoint 5: Output Sanitization."""
    # Normal prose with words like 'tokens' or 'secret' should NOT be blocked
    passed, _ = acs_guard.checkpoint_5_output_sanitization({"output": "Used 150 tokens for secret reasoning"})
    assert passed

    passed, _ = acs_guard.checkpoint_5_output_sanitization({"output": None})
    assert passed

    # Sensitive API key assignment pattern
    passed, reason = acs_guard.checkpoint_5_output_sanitization({"output": "api_key = sk-1234567890"})
    assert not passed
    assert "sensitive pattern" in reason

    # Sensitive RSA private key pattern
    passed, reason = acs_guard.checkpoint_5_output_sanitization({"output": "-----BEGIN RSA PRIVATE KEY-----"})
    assert not passed
    assert "sensitive pattern" in reason

    # Sensitive .env file mention
    passed, reason = acs_guard.checkpoint_5_output_sanitization({"output": "Reading .env file"})
    assert not passed
    assert "sensitive pattern" in reason

    # Tainted output
    passed, reason = acs_guard.checkpoint_5_output_sanitization({"output": mark_tainted("some output")})
    assert not passed
    assert "tainted data detected" in reason


def test_validate_action(acs_guard: ACSGuardRuntimeV7) -> None:
    """Tests validate_action across all 5 checkpoints."""
    valid_data = {
        "action": "read",
        "tool": "ls",
        "current_state": "idle",
        "next_state": "planning",
        "output": "Clean list"
    }
    assert acs_guard.validate_action(valid_data) is True

    invalid_data = dict(valid_data)
    invalid_data["action"] = "unauthorized_action"
    assert acs_guard.validate_action(invalid_data) is False


def test_intercept_action(acs_guard: ACSGuardRuntimeV7, mock_audit_trail: MagicMock, mock_persistence: MagicMock) -> None:
    """Tests intercept_action method."""
    valid_data = {
        "action": "read",
        "tool": "ls",
        "current_state": "idle",
        "next_state": "planning"
    }

    result = acs_guard.intercept_action(valid_data)
    assert result == valid_data
    assert mock_audit_trail.log_call.called
    assert mock_persistence.log_checkpoint.call_count == 5

    # Invalid action should raise ACSRuntimeViolationError (inherits SecurityViolationError)
    invalid_data = dict(valid_data)
    invalid_data["tool"] = "forbidden_tool"

    with pytest.raises(SecurityViolationError) as exc_info:
        acs_guard.intercept_action(invalid_data)

    assert "ACS checkpoint 3" in str(exc_info.value)


def test_execute_with_guard(acs_guard: ACSGuardRuntimeV7) -> None:
    """Tests execute_with_guard for wrapped tool executions."""
    workflow_data = {
        "action": "read",
        "tool": "read_file",
        "current_state": "idle",
        "next_state": "executing",
        "kwargs": {"path": "hello.txt"}
    }

    def dummy_tool(text: str) -> str:
        return f"Content: {text}"

    # Successful guarded execution
    output = acs_guard.execute_with_guard(dummy_tool, workflow_data, "hello world")
    assert output == "Content: hello world"

    # Pre-execution block (invalid action)
    invalid_pre_data = dict(workflow_data)
    invalid_pre_data["action"] = "unauthorized_action"

    with pytest.raises(ACSRuntimeViolationError) as exc_info:
        acs_guard.execute_with_guard(dummy_tool, invalid_pre_data, "hello world")
    assert "Pre-execution blocked" in str(exc_info.value)

    # Post-execution block (sensitive output)
    def leaky_tool() -> str:
        return "api_key = 12345"

    with pytest.raises(ACSRuntimeViolationError) as exc_info:
        acs_guard.execute_with_guard(leaky_tool, workflow_data)
    assert "Post-execution blocked" in str(exc_info.value)
