import pytest
import json
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.marketplace_exporter import MarketplaceExporter

def mock_skill_one(param1: str, param2: int = 42) -> str:
    """A mock skill for testing."""
    return f"{param1} - {param2}"

def mock_skill_two(flag: bool) -> bool:
    """Another mock skill."""
    return not flag

@pytest.fixture
def mock_registry():
    registry = SkillRegistry()
    registry.register_skill("mock_skill_one", mock_skill_one, "A mock skill for testing.")
    registry.register_skill("mock_skill_two", mock_skill_two, "Another mock skill.")
    return registry

def test_export_skills(mock_registry):
    exporter = MarketplaceExporter(mock_registry)
    exported_data = exporter.export_skills()

    assert "skills" in exported_data
    skills = exported_data["skills"]
    assert len(skills) == 2

    skill_one = next((s for s in skills if s["name"] == "mock_skill_one"), None)
    assert skill_one is not None
    assert skill_one["description"] == "A mock skill for testing."
    assert "parameters" in skill_one
    assert skill_one["parameters"]["type"] == "object"
    assert "param1" in skill_one["parameters"]["properties"]
    assert skill_one["parameters"]["properties"]["param1"]["type"] == "string"
    assert "param2" in skill_one["parameters"]["properties"]
    assert skill_one["parameters"]["properties"]["param2"]["type"] == "integer"
    assert "param1" in skill_one["parameters"]["required"]
    assert "param2" not in skill_one["parameters"]["required"]

    skill_two = next((s for s in skills if s["name"] == "mock_skill_two"), None)
    assert skill_two is not None
    assert skill_two["parameters"]["properties"]["flag"]["type"] == "boolean"

def test_export_skills_to_json(mock_registry):
    exporter = MarketplaceExporter(mock_registry)
    json_output = exporter.export_skills_to_json()

    parsed_data = json.loads(json_output)
    assert "skills" in parsed_data
    assert len(parsed_data["skills"]) == 2
    assert parsed_data["skills"][0]["name"] == "mock_skill_one"
