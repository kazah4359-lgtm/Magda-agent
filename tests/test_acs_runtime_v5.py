import pytest
from magda_agent.safety.acs_runtime_v5 import ACSRuntimeGuardV5

class MockPolicyLayer:
    def evaluate(self, tool: str, **kwargs) -> tuple[bool, str]:
        if tool == "mock_safe_tool":
            return True, "Safe."
        if tool == "mock_unsafe_tool":
            return False, "Unsafe operation."
        return False, "Unknown tool."

@pytest.fixture
def guard():
    return ACSRuntimeGuardV5(policy_layer=MockPolicyLayer())

def test_missing_tool(guard):
    passed, reason = guard.evaluate({"action": "execute", "kwargs": {}})
    assert not passed
    assert "missing 'tool' field" in reason

def test_forbidden_tool(guard):
    passed, reason = guard.evaluate({"tool": "forbidden_tool", "kwargs": {}})
    assert not passed
    assert "explicitly forbidden" in reason

def test_invalid_kwargs(guard):
    passed, reason = guard.evaluate({"tool": "mock_safe_tool", "kwargs": "invalid"})
    assert not passed
    assert "'kwargs' must be a dictionary" in reason

def test_safe_tool(guard):
    passed, reason = guard.evaluate({"tool": "mock_safe_tool", "kwargs": {}})
    assert passed
    assert "Tool policy passed" in reason

def test_unsafe_tool(guard):
    passed, reason = guard.evaluate({"tool": "mock_unsafe_tool", "kwargs": {}})
    assert not passed
    assert "Tool policy failed: Unsafe operation." in reason
