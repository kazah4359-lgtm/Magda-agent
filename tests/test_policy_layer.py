import pytest
from typing import Any, Dict, Tuple

from magda_agent.safety.policy_layer import AgentGuardPolicyLayer


def allow_rule(tool_name: str, kwargs: Dict[str, Any]) -> Tuple[bool, str]:
    return True, "Always allowed by rule."


def deny_rule(tool_name: str, kwargs: Dict[str, Any]) -> Tuple[bool, str]:
    return False, "Always denied by rule."


def arg_based_rule(tool_name: str, kwargs: Dict[str, Any]) -> Tuple[bool, str]:
    if kwargs.get("dangerous") is True:
        return False, "Argument 'dangerous' is True."
    return True, "Argument 'dangerous' is not True."


def test_no_rules_fallback_to_base_allow() -> None:
    """Test that with no granular rules, an allowed tool falls back to the base policy and is allowed."""
    policy = AgentGuardPolicyLayer()
    allow, explanation = policy.evaluate("safe_tool", arg="value")

    assert allow is True
    assert "allowed" in explanation.lower()

    trail = policy.get_audit_trail()
    assert len(trail) == 1
    assert trail[0]["tool_name"] == "safe_tool"
    assert trail[0]["result"]["allowed"] is True


def test_no_rules_fallback_to_base_deny() -> None:
    """Test that with no granular rules, a denied tool falls back to the base policy and is denied."""
    policy = AgentGuardPolicyLayer()
    allow, explanation = policy.evaluate("system_execute_code", code="cat .env")

    assert allow is False
    assert "denied" in explanation.lower()


def test_global_rule_deny() -> None:
    """Test that a global rule denying an action takes precedence over the base policy."""
    policy = AgentGuardPolicyLayer()
    policy.add_global_rule(deny_rule)

    allow, explanation = policy.evaluate("safe_tool", arg="value")

    assert allow is False
    assert explanation == "Always denied by rule."

    trail = policy.get_audit_trail()
    assert len(trail) == 1
    assert trail[0]["tool_name"] == "safe_tool"
    assert trail[0]["result"]["allowed"] is False
    assert trail[0]["result"]["explanation"] == explanation


def test_tool_rule_deny() -> None:
    """Test that a tool-specific rule denying an action works for that tool only."""
    policy = AgentGuardPolicyLayer()
    policy.add_tool_rule("specific_tool", deny_rule)

    # specific_tool is denied
    allow1, explanation1 = policy.evaluate("specific_tool", arg="value")
    assert allow1 is False
    assert explanation1 == "Always denied by rule."

    # other_tool falls back to base allow
    allow2, explanation2 = policy.evaluate("other_tool", arg="value")
    assert allow2 is True


def test_tool_rule_alias_resolution() -> None:
    """Test that tool rules are applied correctly for aliases (e.g., programmer -> system_execute_code)."""
    policy = AgentGuardPolicyLayer()
    # Add rule for canonical name
    policy.add_tool_rule("programmer", arg_based_rule)

    # Evaluate using alias 'system_execute_code'
    allow1, explanation1 = policy.evaluate("system_execute_code", dangerous=True)
    assert allow1 is False
    assert explanation1 == "Argument 'dangerous' is True."

    # Evaluate using canonical name
    allow2, _ = policy.evaluate("programmer", dangerous=False)
    assert allow2 is True


def test_rule_evaluation_order() -> None:
    """Test that global rules are evaluated before tool rules."""
    policy = AgentGuardPolicyLayer()

    def global_deny(t: str, k: Dict[str, Any]) -> Tuple[bool, str]:
        return False, "Global deny."

    def tool_allow(t: str, k: Dict[str, Any]) -> Tuple[bool, str]:
        return True, "Tool allow."

    policy.add_global_rule(global_deny)
    policy.add_tool_rule("some_tool", tool_allow)

    allow, explanation = policy.evaluate("some_tool")
    assert allow is False
    assert explanation == "Global deny."
