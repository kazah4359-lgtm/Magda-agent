from magda_agent.safety.policy import PolicyLayer
from magda_agent.skills.registry import SkillRegistry


def test_execute_skill_denied_by_policy() -> None:
    policy = PolicyLayer()
    registry = SkillRegistry(policy_layer=policy)

    def sensitive_skill(code: str = "") -> str:
        return code

    registry.register_skill("system_execute_code", sensitive_skill, "test skill")

    result = registry.execute_skill("system_execute_code", code="open('.env')")

    assert "Action 'system_execute_code' blocked:" in result or result.startswith("Policy denied:")
    assert "denied" in result.lower()


def test_execute_skill_allowed_by_policy() -> None:
    policy = PolicyLayer()
    registry = SkillRegistry(policy_layer=policy)

    def safe_skill(value: str = "") -> str:
        return f"ok:{value}"

    registry.register_skill("safe_skill", safe_skill, "safe")

    result = registry.execute_skill("safe_skill", value="test")

    assert result == "ok:test"
