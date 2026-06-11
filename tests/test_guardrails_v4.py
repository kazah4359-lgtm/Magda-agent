import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from magda_agent.safety.guardrails_v4 import RealtimeGuardrailV4, FallbackStrategyV4, SecurityViolationError
from magda_agent.safety.policy import PolicyLayer

def test_allowed_action() -> None:
    """Tests that an allowed action executes normally."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(True, "Allowed"))
    guard = RealtimeGuardrailV4(policy)
    mock_tool = MagicMock(return_value="Success")

    result = guard.execute_with_guardrails(mock_tool, "safe_tool", arg="value")
    assert result == "Success"
    mock_tool.assert_called_once_with(arg="value")

def test_denied_action_stop_execution() -> None:
    """Tests default STOP_EXECUTION on policy violation."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(False, "Denied"))
    guard = RealtimeGuardrailV4(policy, default_strategy=FallbackStrategyV4.STOP_EXECUTION)
    mock_tool = MagicMock()

    with pytest.raises(SecurityViolationError, match="Action 'unsafe_tool' blocked: Denied"):
        guard.execute_with_guardrails(mock_tool, "unsafe_tool", arg="value")

    mock_tool.assert_not_called()

def test_dynamic_handler_sync_fallback() -> None:
    """Tests that a sync dynamic fallback handler is called on violation."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(False, "Policy issue"))
    guard = RealtimeGuardrailV4(policy)

    fallback_mock = MagicMock(return_value="Handled Sync")
    guard.register_fallback_handler("risky_tool", fallback_mock)

    mock_tool = MagicMock()
    result = guard.execute_with_guardrails(mock_tool, "risky_tool", arg="value")

    assert result == "Handled Sync"
    fallback_mock.assert_called_once_with("risky_tool", "Policy issue", arg="value")
    mock_tool.assert_not_called()

@pytest.mark.asyncio
async def test_dynamic_handler_async_fallback() -> None:
    """Tests that an async dynamic fallback handler is called on violation."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(False, "Async Policy issue"))
    guard = RealtimeGuardrailV4(policy)

    async def async_fallback(tool_name: str, explanation: str, **kwargs) -> str:
        return f"Handled Async for {tool_name}"

    guard.register_fallback_handler("async_risky_tool", async_fallback)

    async def mock_async_tool(arg: str) -> str:
        return "Async Success"

    result_coro = guard.execute_with_guardrails(mock_async_tool, "async_risky_tool", arg="value")
    assert asyncio.iscoroutine(result_coro)
    result = await result_coro

    assert result == "Handled Async for async_risky_tool"

def test_exception_graceful_degradation_sync() -> None:
    """Tests graceful degradation via dynamic handler on synchronous tool exception."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(True, "Allowed"))
    guard = RealtimeGuardrailV4(policy)

    fallback_mock = MagicMock(return_value="Recovered Sync Exception")
    guard.register_fallback_handler("failing_tool", fallback_mock)

    mock_tool = MagicMock(side_effect=ValueError("Test Sync Error"))

    result = guard.execute_with_guardrails(mock_tool, "failing_tool", arg="value")

    assert result == "Recovered Sync Exception"
    mock_tool.assert_called_once_with(arg="value")
    fallback_mock.assert_called_once_with("failing_tool", "Exception during execution: Test Sync Error", arg="value")

@pytest.mark.asyncio
async def test_exception_graceful_degradation_async() -> None:
    """Tests graceful degradation via dynamic handler on async tool exception."""
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(True, "Allowed"))
    guard = RealtimeGuardrailV4(policy)

    async def async_fallback(tool_name: str, explanation: str, **kwargs) -> str:
        return "Recovered Async Exception"

    guard.register_fallback_handler("async_failing_tool", async_fallback)

    async def failing_async_tool(arg: str) -> str:
        raise ValueError("Async Test Error")

    result_coro = guard.execute_with_guardrails(failing_async_tool, "async_failing_tool", arg="value")
    assert asyncio.iscoroutine(result_coro)
    result = await result_coro

    assert result == "Recovered Async Exception"
