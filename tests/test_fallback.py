import pytest
from typing import Any
from magda_agent.safety.fallback import RealtimeGuardrailFallback
from magda_agent.safety.policy import PolicyLayer

def mock_tool(**kwargs: Any) -> str:
    """
    Mock tool function for testing.
    """
    return "success"

def mock_fallback(explanation: str, **kwargs: Any) -> str:
    """
    Mock fallback handler.
    """
    return f"fallback triggered: {explanation}"

class MockPolicyLayer(PolicyLayer):
    """
    Mock PolicyLayer for testing.
    """
    def evaluate(self, tool_name: str, **kwargs: Any) -> tuple[bool, str]:
        """
        Mocks evaluate method.
        """
        if tool_name == "blocked_tool":
            return False, "This tool is blocked"
        return True, "Allowed"

def test_fallback_success() -> None:
    """
    Tests successful execution when tool is allowed.
    """
    fallback = RealtimeGuardrailFallback(policy_layer=MockPolicyLayer())
    result = fallback.execute_with_fallback(mock_tool, "allowed_tool", arg="test")
    assert result == "success"

def test_fallback_triggered() -> None:
    """
    Tests that a fallback handler is executed when a tool is blocked.
    """
    fallback = RealtimeGuardrailFallback(policy_layer=MockPolicyLayer())
    fallback.register_fallback("blocked_tool", mock_fallback)
    result = fallback.execute_with_fallback(mock_tool, "blocked_tool", arg="test")
    assert "fallback triggered: This tool is blocked" in result

def test_fallback_no_handler() -> None:
    """
    Tests that a ValueError is raised when a tool is blocked and no handler exists.
    """
    fallback = RealtimeGuardrailFallback(policy_layer=MockPolicyLayer())
    with pytest.raises(ValueError, match="Action blocked by policy and no fallback available"):
        fallback.execute_with_fallback(mock_tool, "blocked_tool", arg="test")
