"""
Tests for the Agent Guard module.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from magda_agent.safety.agent_guard import AgentGuard, SecurityViolationError
from magda_agent.safety.policy import PolicyLayer

@pytest.fixture
def policy_layer_mock():
    mock = MagicMock(spec=PolicyLayer)
    # Default behavior: allow everything
    mock.evaluate.return_value = (True, "Allowed by default mock.")
    return mock

@pytest.fixture
def agent_guard(policy_layer_mock):
    return AgentGuard(policy_layer=policy_layer_mock)

def test_agent_guard_allows_tool_execution(agent_guard, policy_layer_mock):
    # Setup
    tool_mock = MagicMock(return_value="tool_result")

    # Execute
    result = agent_guard.execute_tool(tool_mock, "test_tool", arg1="value1")

    # Verify
    policy_layer_mock.evaluate.assert_called_once_with("test_tool", arg1="value1")
    tool_mock.assert_called_once_with(arg1="value1")
    assert result == "tool_result"

@pytest.mark.asyncio
async def test_agent_guard_allows_async_tool_execution(agent_guard, policy_layer_mock):
    # Setup
    async def async_tool(arg1):
        return f"async_{arg1}"

    # Execute
    result = await agent_guard.execute_tool(async_tool, "test_async_tool", arg1="value1")

    # Verify
    policy_layer_mock.evaluate.assert_called_once_with("test_async_tool", arg1="value1")
    assert result == "async_value1"

def test_agent_guard_blocks_tool_execution(agent_guard, policy_layer_mock):
    # Setup
    policy_layer_mock.evaluate.return_value = (False, "Blocked for testing.")
    tool_mock = MagicMock()

    # Execute and Verify Exception
    with pytest.raises(SecurityViolationError) as exc_info:
        agent_guard.execute_tool(tool_mock, "dangerous_tool", secret="123")

    assert "Blocked for testing." in str(exc_info.value)
    policy_layer_mock.evaluate.assert_called_once_with("dangerous_tool", secret="123")
    # Verify the tool was NOT called
    tool_mock.assert_not_called()

def test_agent_guard_logging(agent_guard, policy_layer_mock, caplog):
    import logging
    caplog.set_level(logging.INFO)

    tool_mock = MagicMock(return_value="ok")

    # Test permitted logging
    agent_guard.execute_tool(tool_mock, "safe_tool")
    assert "AgentGuard: Tool execution permitted for 'safe_tool'." in caplog.text

    caplog.clear()

    # Test blocked logging
    policy_layer_mock.evaluate.return_value = (False, "Policy deny.")
    with pytest.raises(SecurityViolationError):
        agent_guard.execute_tool(tool_mock, "unsafe_tool")

    assert "AgentGuard: Tool execution blocked for 'unsafe_tool'. Reason: Policy deny." in caplog.text

def test_agent_guard_decorator(agent_guard, policy_layer_mock):
    @agent_guard.guard_tool("decorated_tool")
    def my_tool(kwarg1=None):
        return "success"

    result = my_tool(kwarg1="test")
    policy_layer_mock.evaluate.assert_called_once_with("decorated_tool", kwarg1="test")
    assert result == "success"

@pytest.mark.asyncio
async def test_agent_guard_async_decorator(agent_guard, policy_layer_mock):
    @agent_guard.guard_tool("async_decorated_tool")
    async def my_async_tool(kwarg1=None):
        return "async_success"

    result = await my_async_tool(kwarg1="test")
    policy_layer_mock.evaluate.assert_called_once_with("async_decorated_tool", kwarg1="test")
    assert result == "async_success"
