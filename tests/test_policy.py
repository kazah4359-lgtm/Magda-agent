import pytest
from magda_agent.safety.policy import PolicyLayer
from magda_agent.action.selector import BasalGanglia

def test_policy_layer_allow() -> None:
    """Test that a safe tool is allowed by the policy layer."""
    policy = PolicyLayer()
    allow, explanation = policy.evaluate("safe_tool", arg1="value")
    assert allow is True
    assert "allowed" in explanation.lower()

    trail = policy.get_audit_trail()
    assert len(trail) == 1
    assert trail[0]["tool_name"] == "safe_tool"
    assert trail[0]["kwargs"] == {"arg1": "value"}
    assert trail[0]["result"]["allowed"] is True

def test_policy_layer_deny_system_execute_code() -> None:
    """Test that system_execute_code is denied when trying to access sensitive paths."""
    policy = PolicyLayer()
    # Mocking a code execution that touches .env
    allow, explanation = policy.evaluate("system_execute_code", code="cat .env")
    assert allow is False
    assert "denied" in explanation.lower()

    trail = policy.get_audit_trail()
    assert len(trail) == 1
    assert trail[0]["tool_name"] == "system_execute_code"
    assert trail[0]["result"]["allowed"] is False

def test_policy_layer_deny_send_message() -> None:
    """Test that sending a message to a blocked user is denied."""
    policy = PolicyLayer()
    allow, explanation = policy.evaluate("send_message", recipient="blocked_user", message="hello")
    assert allow is False
    assert "denied" in explanation.lower()

def test_basal_ganglia_with_policy() -> None:
    """Test that BasalGanglia filters out denied actions and selects the next best allowed action."""
    policy = PolicyLayer()
    bg = BasalGanglia(policy_layer=policy)

    actions = [
        {"action": "system_execute_code", "priority": 10, "kwargs": {"code": "cat .env"}}, # Should be denied
        {"action": "safe_tool", "priority": 5, "kwargs": {"param": "value"}},              # Should be allowed
        {"action": "another_safe_tool", "priority": 1, "kwargs": {}}                       # Allowed but lower priority
    ]

    selected = bg.select_action(actions)

    # The highest priority is denied, so it should select the second highest
    assert selected is not None
    assert selected["action"] == "safe_tool"
    assert selected["priority"] == 5

def test_basal_ganglia_all_denied() -> None:
    """Test that BasalGanglia returns None if all actions are denied by the policy layer."""
    policy = PolicyLayer()
    bg = BasalGanglia(policy_layer=policy)

    actions = [
        {"action": "system_execute_code", "priority": 10, "kwargs": {"code": "cat secrets/keys"}},
        {"action": "send_message", "priority": 5, "kwargs": {"recipient": "blocked_user"}}
    ]

    selected = bg.select_action(actions)
    assert selected is None

def test_policy_layer_deny_programmer_alias() -> None:
    policy = PolicyLayer()
    allow, explanation = policy.evaluate("programmer", code="open('.env').read()")
    assert allow is False
    assert "denied" in explanation.lower()


def test_policy_layer_deny_omnichannel_alias() -> None:
    policy = PolicyLayer()
    allow, explanation = policy.evaluate("omnichannel_send", recipient="blocked_user", message="hello")
    assert allow is False
    assert "denied" in explanation.lower()
