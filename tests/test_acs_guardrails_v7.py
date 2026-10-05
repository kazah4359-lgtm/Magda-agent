import pytest
from unittest.mock import MagicMock, AsyncMock

from magda_agent.safety.acs_guardrails_v7 import (
    ACSGuardrailsV7,
    ACSGuardrails,
    ACSCompliancePolicyGuardrailsV7,
    ACSGuardrailViolationError
)
from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.taint import mark_tainted


@pytest.fixture
def mock_policy_layer():
    policy = MagicMock(spec=PolicyLayer)
    policy.evaluate.return_value = (True, "Allowed by mock policy")
    return policy


@pytest.fixture
def mock_audit_trail():
    return MagicMock()


@pytest.fixture
def mock_persistence():
    return MagicMock()


@pytest.fixture
def guardrails(mock_policy_layer, mock_audit_trail, mock_persistence):
    return ACSGuardrailsV7(
        policy_layer=mock_policy_layer,
        audit_trail=mock_audit_trail,
        persistence=mock_persistence
    )


def test_aliases():
    """Verify class aliases match ACSGuardrailsV7."""
    assert ACSGuardrails is ACSGuardrailsV7
    assert ACSCompliancePolicyGuardrailsV7 is ACSGuardrailsV7


def test_validate_policy_success(guardrails, mock_policy_layer):
    """Test successful policy validation for valid inputs."""
    passed, reason, details = guardrails.validate_policy("read_file", {"filepath": "doc.txt"})
    assert passed is True
    assert "passed" in reason.lower()
    assert details["tool"] == "read_file"
    mock_policy_layer.evaluate.assert_called_once_with("read_file", filepath="doc.txt")


def test_validate_policy_invalid_tool_name(guardrails):
    """Test policy validation rejection for invalid tool names."""
    passed, reason, details = guardrails.validate_policy("", {"filepath": "doc.txt"})
    assert passed is False
    assert "tool name must be a non-empty string" in reason

    passed_none, reason_none, _ = guardrails.validate_policy(None, {})
    assert passed_none is False


def test_validate_policy_tainted_input(guardrails):
    """Test policy validation rejection for tainted tool name or kwargs."""
    tainted_tool = mark_tainted("read_file")
    passed, reason, _ = guardrails.validate_policy(tainted_tool, {})
    assert passed is False
    assert "tainted" in reason

    tainted_kwargs = mark_tainted({"cmd": "rm -rf /"})
    passed_kw, reason_kw, _ = guardrails.validate_policy("exec", tainted_kwargs)
    assert passed_kw is False
    assert "tainted" in reason_kw


def test_validate_policy_central_denial(guardrails, mock_policy_layer):
    """Test policy validation when PolicyLayer denies the action."""
    mock_policy_layer.evaluate.return_value = (False, "Operation restricted")
    passed, reason, details = guardrails.validate_policy("write_file", {"filepath": "/etc/passwd"})
    assert passed is False
    assert "Operation restricted" in reason


def test_custom_policy_rules(guardrails):
    """Test registering and enforcing custom sync policy rules."""
    def disallow_delete(tool_name, kwargs, context):
        if tool_name.startswith("delete"):
            return False, "Deletion is disabled by custom rule"
        return True, "OK"

    guardrails.add_policy_rule(disallow_delete)

    passed, reason, _ = guardrails.validate_policy("read_file", {})
    assert passed is True

    passed_del, reason_del, details_del = guardrails.validate_policy("delete_file", {"path": "tmp.txt"})
    assert passed_del is False
    assert "Deletion is disabled by custom rule" in reason_del


@pytest.mark.asyncio
async def test_validate_policy_async_with_async_mock(guardrails):
    """Test async policy validation using AsyncMock for custom policy callbacks."""
    async_rule = AsyncMock(return_value=(False, "Async rule violation"))
    guardrails.add_policy_rule(async_rule)

    passed, reason, details = await guardrails.validate_policy_async("custom_tool", {"arg": "val"})
    assert passed is False
    assert "Async rule violation" in reason
    async_rule.assert_awaited_once_with("custom_tool", {"arg": "val"}, {})


def test_execute_with_guardrails_success(guardrails, mock_audit_trail):
    """Test successful synchronous tool execution wrapped with guardrails."""
    def dummy_tool(text: str) -> str:
        return f"Hello, {text}!"

    result = guardrails.execute_with_guardrails(
        tool_name="dummy_tool",
        kwargs={"text": "World"},
        tool_func=dummy_tool
    )
    assert result == "Hello, World!"
    assert mock_audit_trail.log_call.called


def test_execute_with_guardrails_pre_execution_blocked(guardrails, mock_policy_layer):
    """Test execute_with_guardrails raising ACSGuardrailViolationError on pre-execution failure."""
    mock_policy_layer.evaluate.return_value = (False, "Forbidden tool execution")
    tool_mock = MagicMock()

    with pytest.raises(ACSGuardrailViolationError) as exc_info:
        guardrails.execute_with_guardrails(
            tool_name="forbidden_action",
            kwargs={},
            tool_func=tool_mock
        )

    assert "Pre-execution policy violation" in str(exc_info.value)
    tool_mock.assert_not_called()


def test_execute_with_guardrails_output_sanitization_blocked(guardrails):
    """Test execute_with_guardrails blocking sensitive outputs."""
    def leaky_tool() -> str:
        return "api_key = sk-1234567890abcdef"

    with pytest.raises(ACSGuardrailViolationError) as exc_info:
        guardrails.execute_with_guardrails(
            tool_name="get_key",
            kwargs={},
            tool_func=leaky_tool
        )

    assert "Post-execution output sanitization violation" in str(exc_info.value)


@pytest.mark.asyncio
async def test_execute_with_guardrails_async_success(guardrails):
    """Test async tool execution with AsyncMock passing guardrails."""
    async_tool_mock = AsyncMock(return_value="Async execution result")

    result = await guardrails.execute_with_guardrails_async(
        tool_name="async_tool",
        kwargs={"param": 42},
        tool_func=async_tool_mock
    )

    assert result == "Async execution result"
    async_tool_mock.assert_awaited_once_with(param=42)


@pytest.mark.asyncio
async def test_execute_with_guardrails_async_pre_execution_blocked(guardrails, mock_policy_layer):
    """Test async tool execution blocked prior to action via AsyncMock."""
    mock_policy_layer.evaluate.return_value = (False, "Blocked by central policy")
    async_tool_mock = AsyncMock()

    with pytest.raises(ACSGuardrailViolationError) as exc_info:
        await guardrails.execute_with_guardrails_async(
            tool_name="blocked_async_tool",
            kwargs={"data": "test"},
            tool_func=async_tool_mock
        )

    assert "Pre-execution policy violation" in str(exc_info.value)
    async_tool_mock.assert_not_called()


@pytest.mark.asyncio
async def test_execute_with_guardrails_async_output_sanitization_blocked(guardrails):
    """Test async tool execution blocked by output sanitization using AsyncMock."""
    async_leaky_mock = AsyncMock(return_value="AWS_SECRET_ACCESS_KEY = 12345678")

    with pytest.raises(ACSGuardrailViolationError) as exc_info:
        await guardrails.execute_with_guardrails_async(
            tool_name="leaky_async_tool",
            kwargs={},
            tool_func=async_leaky_mock
        )

    assert "Post-execution output sanitization violation" in str(exc_info.value)
    async_leaky_mock.assert_awaited_once()
