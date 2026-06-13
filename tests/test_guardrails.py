import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from magda_agent.safety.guardrails import RealtimeGuardrail, FallbackStrategy, SecurityViolationError
from magda_agent.safety.policy import PolicyLayer

def test_allowed_action() -> None:
    """Tests that an allowed action is executed and returns the result."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(True, "Allowed"))
    guard = RealtimeGuardrail(policy)
    mock_tool = MagicMock(return_value="Success")

    result = guard.execute_with_guardrails(mock_tool, "safe_tool", arg="value")
    assert result == "Success"
    mock_tool.assert_called_once_with(arg="value")

def test_denied_action_stop_execution() -> None:
    """Tests that a denied action with STOP_EXECUTION strategy raises SecurityViolationError."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(False, "Denied"))
    guard = RealtimeGuardrail(policy, default_strategy=FallbackStrategy.STOP_EXECUTION)
    mock_tool = MagicMock()

    with pytest.raises(SecurityViolationError, match="Action 'unsafe_tool' blocked: Denied"):
        guard.execute_with_guardrails(mock_tool, "unsafe_tool", arg="value")

    mock_tool.assert_not_called()

def test_denied_action_request_review() -> None:
    """Tests that a denied action with REQUEST_REVIEW strategy returns a review message."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(False, "Needs Review"))
    guard = RealtimeGuardrail(policy, default_strategy=FallbackStrategy.REQUEST_REVIEW)
    mock_tool = MagicMock()

    result = guard.execute_with_guardrails(mock_tool, "unsafe_tool", arg="value")
    assert result == "Review requested for action 'unsafe_tool': Needs Review"
    mock_tool.assert_not_called()

@pytest.mark.asyncio
async def test_denied_action_async_tool() -> None:
    """Tests that an async tool fallback is handled correctly and returns an awaitable."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(False, "Needs Review"))
    guard = RealtimeGuardrail(policy, default_strategy=FallbackStrategy.REQUEST_REVIEW)

    async def async_mock_tool(arg: str) -> str:
        return "Async Success"

    result_coro = guard.execute_with_guardrails(async_mock_tool, "unsafe_tool", arg="value")
    assert asyncio.iscoroutine(result_coro)
    result = await result_coro
    assert result == "Review requested for action 'unsafe_tool': Needs Review"


def test_denied_action_warn_and_continue() -> None:
    """Tests that a denied action with WARN_AND_CONTINUE strategy returns a warning message."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(False, "Warning Reason"))
    guard = RealtimeGuardrail(policy, default_strategy=FallbackStrategy.WARN_AND_CONTINUE)
    mock_tool = MagicMock()

    result = guard.execute_with_guardrails(mock_tool, "unsafe_tool", arg="value")
    assert result == "Warning: Warning Reason. Action 'unsafe_tool' skipped."
    mock_tool.assert_not_called()


def test_allowed_action_exception_fallback() -> None:
    """Tests that an allowed action handles exceptions and returns a fallback message."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(True, "Allowed"))
    guard = RealtimeGuardrail(policy)
    mock_tool = MagicMock(side_effect=ValueError("Test Error"))

    result = guard.execute_with_guardrails(mock_tool, "failing_tool", arg="value")
    assert result == "Action 'failing_tool' failed during execution: Test Error"
    mock_tool.assert_called_once_with(arg="value")

@pytest.mark.asyncio
async def test_allowed_action_async_exception_fallback() -> None:
    """Tests that an allowed async action handles exceptions and returns a fallback message."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(True, "Allowed"))
    guard = RealtimeGuardrail(policy)

    async def failing_async_tool(arg: str) -> str:
        raise ValueError("Async Test Error")

    result_coro = guard.execute_with_guardrails(failing_async_tool, "failing_async_tool", arg="value")
    assert asyncio.iscoroutine(result_coro)
    result = await result_coro
    assert result == "Action 'failing_async_tool' failed during execution: Async Test Error"

@pytest.mark.asyncio
async def test_allowed_action_async_interrupted_fallback() -> None:
    """Tests that an allowed async action handles CancelledError properly by re-raising."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(True, "Allowed"))
    guard = RealtimeGuardrail(policy)

    async def interrupted_async_tool(arg: str) -> str:
        raise asyncio.CancelledError()

    result_coro = guard.execute_with_guardrails(interrupted_async_tool, "interrupted_async_tool", arg="value")
    assert asyncio.iscoroutine(result_coro)
    with pytest.raises(asyncio.CancelledError):
        await result_coro

def test_allowed_action_interrupted_fallback() -> None:
    """Tests that an allowed sync action handles KeyboardInterrupt properly by re-raising."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(True, "Allowed"))
    guard = RealtimeGuardrail(policy)

    def interrupted_tool(arg: str) -> str:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        guard.execute_with_guardrails(interrupted_tool, "interrupted_tool", arg="value")
