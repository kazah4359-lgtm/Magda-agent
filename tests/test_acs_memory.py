import pytest

from magda_agent.safety.acs_memory import ACSMemoryPolicy


def test_acs_memory_policy_allowed_read():
    """Tests that read_memory with a matching user_id passes evaluation."""
    policy = ACSMemoryPolicy(expected_user_id="user_123")
    allow, explanation = policy.evaluate("read_memory", user_id="user_123")
    assert allow is True
    assert "allowed" in explanation.lower()


def test_acs_memory_policy_allowed_write():
    """Tests that write_memory with a matching user_id passes evaluation."""
    policy = ACSMemoryPolicy(expected_user_id="user_456")
    allow, explanation = policy.evaluate("write_memory", user_id="user_456")
    assert allow is True
    assert "allowed" in explanation.lower()


def test_acs_memory_policy_denied_read_no_user():
    """Tests that read_memory without a user_id fails evaluation."""
    policy = ACSMemoryPolicy(expected_user_id="user_123")
    allow, explanation = policy.evaluate("read_memory")
    assert allow is False
    assert "denied" in explanation.lower()
    assert "missing user_id" in explanation.lower()


def test_acs_memory_policy_denied_write_mismatch_user():
    """Tests that write_memory with a mismatched user_id fails evaluation."""
    policy = ACSMemoryPolicy(expected_user_id="user_123")
    allow, explanation = policy.evaluate("write_memory", user_id="hacker_999")
    assert allow is False
    assert "denied" in explanation.lower()
    assert "user_id mismatch" in explanation.lower()


def test_acs_memory_policy_delegates_non_memory_tools():
    """Tests that non-memory tool calls are delegated to the base PolicyLayer."""
    policy = ACSMemoryPolicy()
    allow, explanation = policy.evaluate("some_other_tool", param="value")
    assert allow is True
    assert "allowed" in explanation.lower()
