import pytest
import json
from typing import List, Dict, Any
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.marketplace_exporter_v3 import MarketplaceExporterV3


def mock_skill_one(param1: str, param2: int = 42) -> str:
    """A mock skill for testing."""
    return f"{param1} - {param2}"


def mock_skill_two(flag: bool, tags: List[str]) -> bool:
    """Another mock skill with list."""
    return not flag


def mock_skill_three(config: Dict[str, Any]) -> str:
    """Mock skill with dict."""
    return str(config)


@pytest.fixture
def mock_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register_skill("mock_skill_one", mock_skill_one, "A mock skill for testing.")
    registry.register_skill("mock_skill_two", mock_skill_two, "Another mock skill with list.")
    registry.register_skill("mock_skill_three", mock_skill_three, "Mock skill with dict.")
    return registry


def test_export_skills(mock_registry: SkillRegistry) -> None:
    exporter = MarketplaceExporterV3(mock_registry)
    exported_data = exporter.export_skills()

    assert "skills" in exported_data
    skills = exported_data["skills"]
    assert len(skills) == 3

    # Test skill one
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

    # Test skill two
    skill_two = next((s for s in skills if s["name"] == "mock_skill_two"), None)
    assert skill_two is not None
    assert skill_two["parameters"]["properties"]["flag"]["type"] == "boolean"
    assert skill_two["parameters"]["properties"]["tags"]["type"] == "array"
    assert skill_two["parameters"]["properties"]["tags"]["items"]["type"] == "string"
    assert "flag" in skill_two["parameters"]["required"]
    assert "tags" in skill_two["parameters"]["required"]

    # Test skill three
    skill_three = next((s for s in skills if s["name"] == "mock_skill_three"), None)
    assert skill_three is not None
    assert skill_three["parameters"]["properties"]["config"]["type"] == "object"
    assert "config" in skill_three["parameters"]["required"]


def test_export_skills_to_json(mock_registry: SkillRegistry) -> None:
    exporter = MarketplaceExporterV3(mock_registry)
    json_output = exporter.export_skills_to_json()

    parsed_data = json.loads(json_output)
    assert "skills" in parsed_data
    assert len(parsed_data["skills"]) == 3
    assert parsed_data["skills"][0]["name"] == "mock_skill_one"
