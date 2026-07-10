import pytest
import asyncio
from unittest.mock import MagicMock
from magda_agent.safety.policy import PolicyLayer
from magda_agent.safety.guardrails import FallbackStrategy, SecurityViolationError
from magda_agent.safety.guard_runtime_policy_v5 import AgentGuardRuntimePolicyV5, DynamicRule

class AlwaysDenyRule(DynamicRule):
    def evaluate(self, tool_name: str, **kwargs) -> tuple:
        return False, "Denied by dynamic rule"

class AlwaysAllowRule(DynamicRule):
    def evaluate(self, tool_name: str, **kwargs) -> tuple:
        return True, "Allowed by dynamic rule"

def test_dynamic_rule_blocks_action() -> None:
    """Tests that a dynamic rule correctly blocks an action."""
    mock_policy = MagicMock(spec=PolicyLayer)
    # The static policy would allow it, but dynamic should block it first
    mock_policy.evaluate.return_value = (True, "Allowed by static")

    guard = AgentGuardRuntimePolicyV5(mock_policy)
    guard.add_dynamic_rule(AlwaysDenyRule())

    mock_tool = MagicMock()

    with pytest.raises(SecurityViolationError, match="Action 'test_tool' blocked: Denied by dynamic rule"):
        guard.execute_with_guardrails(mock_tool, "test_tool")

    mock_tool.assert_not_called()
    # Static policy shouldn't even be reached if dynamic blocks
    mock_policy.evaluate.assert_not_called()

def test_static_policy_blocks_action_after_dynamic_allows() -> None:
    """Tests that the static policy can block an action even if dynamic rules allow it."""
    mock_policy = MagicMock(spec=PolicyLayer)
    mock_policy.evaluate.return_value = (False, "Blocked by static")

    guard = AgentGuardRuntimePolicyV5(mock_policy)
    guard.add_dynamic_rule(AlwaysAllowRule())

    mock_tool = MagicMock()

    with pytest.raises(SecurityViolationError, match="Action 'test_tool' blocked: Blocked by static"):
        guard.execute_with_guardrails(mock_tool, "test_tool")

    mock_tool.assert_not_called()
    mock_policy.evaluate.assert_called_once_with("test_tool")

def test_allowed_action_executes() -> None:
    """Tests that an action is executed when both dynamic and static policies allow it."""
    mock_policy = MagicMock(spec=PolicyLayer)
    mock_policy.evaluate.return_value = (True, "Allowed by static")

    guard = AgentGuardRuntimePolicyV5(mock_policy)
    guard.add_dynamic_rule(AlwaysAllowRule())

    mock_tool = MagicMock(return_value="Success")

    result = guard.execute_with_guardrails(mock_tool, "safe_tool", arg="value")
    assert result == "Success"
    mock_tool.assert_called_once_with(arg="value")
    mock_policy.evaluate.assert_called_once_with("safe_tool", arg="value")

@pytest.mark.asyncio
async def test_allowed_action_async_tool() -> None:
    """Tests that an async tool executes when allowed."""
    mock_policy = MagicMock(spec=PolicyLayer)
    mock_policy.evaluate.return_value = (True, "Allowed by static")

    guard = AgentGuardRuntimePolicyV5(mock_policy)
    guard.add_dynamic_rule(AlwaysAllowRule())

    async def async_mock_tool(arg: str) -> str:
        return "Async Success"

    result_coro = guard.execute_with_guardrails(async_mock_tool, "safe_async_tool", arg="value")
    assert asyncio.iscoroutine(result_coro)
    result = await result_coro
    assert result == "Async Success"
    mock_policy.evaluate.assert_called_once_with("safe_async_tool", arg="value")
