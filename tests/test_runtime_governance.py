"""
Tests for the Runtime Governance Layer module.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from magda_agent.safety.runtime_governance import RuntimeGovernanceLayer, GovernanceViolationError
from magda_agent.safety.policy import PolicyLayer

@pytest.fixture
def policy_layer_mock() -> MagicMock:
    """Provides a MagicMock for PolicyLayer."""
    mock = MagicMock(spec=PolicyLayer)
    # Default mock allows execution
    mock.evaluate.return_value = (True, "Allowed mock reason.")
    return mock

@pytest.fixture
def governance_layer(policy_layer_mock: MagicMock) -> RuntimeGovernanceLayer:
    """Provides an instance of RuntimeGovernanceLayer."""
    return RuntimeGovernanceLayer(policy_layer=policy_layer_mock)

def test_governance_layer_allows_execution(governance_layer: RuntimeGovernanceLayer, policy_layer_mock: MagicMock) -> None:
    """Test that execution is allowed and function is called with correct arguments."""
    tool_mock = MagicMock(return_value="mock_result")

    result = governance_layer.execute_tool(tool_mock, "test_tool", my_arg="value")

    policy_layer_mock.evaluate.assert_called_once_with("test_tool", my_arg="value")
    tool_mock.assert_called_once_with(my_arg="value")
    assert result == "mock_result"

@pytest.mark.asyncio
async def test_governance_layer_allows_async_execution(governance_layer: RuntimeGovernanceLayer, policy_layer_mock: MagicMock) -> None:
    """Test that async execution is allowed and correctly awaited."""
    async def dummy_tool(my_arg):
        return f"async_{my_arg}"

    result = await governance_layer.execute_tool(dummy_tool, "test_async_tool", my_arg="value")

    policy_layer_mock.evaluate.assert_called_once_with("test_async_tool", my_arg="value")
    assert result == "async_value"

def test_governance_layer_blocks_execution(governance_layer: RuntimeGovernanceLayer, policy_layer_mock: MagicMock) -> None:
    """Test that execution is blocked when policy denies it."""
    # Setup mock to deny execution
    policy_layer_mock.evaluate.return_value = (False, "Blocked by policy.")
    tool_mock = MagicMock()

    with pytest.raises(GovernanceViolationError) as exc_info:
        governance_layer.execute_tool(tool_mock, "unsafe_tool", secret_data="123")

    assert "Blocked by policy" in str(exc_info.value)
    policy_layer_mock.evaluate.assert_called_once_with("unsafe_tool", secret_data="123")
    # Verify tool was completely blocked
    tool_mock.assert_not_called()

def test_governance_guard_decorator(governance_layer: RuntimeGovernanceLayer, policy_layer_mock: MagicMock) -> None:
    """Test the decorator allows execution."""
    @governance_layer.governance_guard("decorated_tool")
    def sample_tool(arg_test=None):
        return "success_decor"

    result = sample_tool(arg_test="ok")
    policy_layer_mock.evaluate.assert_called_once_with("decorated_tool", arg_test="ok")
    assert result == "success_decor"

@pytest.mark.asyncio
async def test_governance_guard_async_decorator(governance_layer: RuntimeGovernanceLayer, policy_layer_mock: MagicMock) -> None:
    """Test the async decorator allows execution."""
    @governance_layer.governance_guard("async_decorated_tool")
    async def sample_async_tool(arg_test=None):
        return "async_success_decor"

    result = await sample_async_tool(arg_test="ok")
    policy_layer_mock.evaluate.assert_called_once_with("async_decorated_tool", arg_test="ok")
    assert result == "async_success_decor"
