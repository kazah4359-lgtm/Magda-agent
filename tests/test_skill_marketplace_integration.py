import pytest
from magda_agent.skills.registry import SkillRegistry
from magda_agent.integration.skill_marketplace import AgentSkillsExporter

def mock_skill(query: str, max_results: int = 5) -> str:
    """A mock skill for testing."""
    return f"Results for {query}"

def test_agent_skills_exporter_format() -> None:
    """Tests the format of exported agentskills.io schemas."""
    registry = SkillRegistry()
    registry.register_skill("mock_skill", mock_skill, "A mock skill for testing.")
    exporter = AgentSkillsExporter(registry)
    exported = exporter.export_skills()

    assert len(exported) == 1
    skill = exported[0]

    assert skill["name"] == "mock_skill"
    assert skill["description"] == "A mock skill for testing."
    assert "parameters" in skill

    params = skill["parameters"]
    assert params["type"] == "object"
    assert "query" in params["properties"]
    assert params["properties"]["query"]["type"] == "string"
    assert "max_results" in params["properties"]
    assert params["properties"]["max_results"]["type"] == "integer"
    assert "query" in params["required"]
    assert "max_results" not in params["required"]
