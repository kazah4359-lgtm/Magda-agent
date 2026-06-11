from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.agentskills_export import AgentSkillsExporter

def mock_skill(param_one: str, param_two: int = 42) -> str:
    """A mock skill for testing."""
    return f"{param_one} {param_two}"

def test_agentskills_exporter_schema_validation():
    registry = SkillRegistry()
    registry.register_skill("mock_skill", mock_skill, "A mock skill for testing.")

    exporter = AgentSkillsExporter(registry)
    skills = exporter.export_skills()

    assert len(skills) == 1
    skill = skills[0]

    assert skill["name"] == "mock_skill"
    assert skill["description"] == "A mock skill for testing."
    assert "parameters" in skill

    params = skill["parameters"]
    assert params["type"] == "object"
    assert "param_one" in params["properties"]
    assert params["properties"]["param_one"]["type"] == "string"
    assert "param_two" in params["properties"]
    assert params["properties"]["param_two"]["type"] == "integer"

    assert "param_one" in params["required"]
    assert "param_two" not in params["required"]
