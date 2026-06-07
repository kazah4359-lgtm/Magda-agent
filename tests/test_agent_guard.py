"""
Tests for the Agent Guard module.
"""

import pytest
from unittest.mock import MagicMock
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
