import json
import pytest
from typing import Dict, Any, List
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.agentskills_importer import AgentSkillsImporter

def test_importer_single_skill_dict() -> None:
    """Tests importing a single skill defined as a dictionary."""
    importer = AgentSkillsImporter()
    skill_def: Dict[str, Any] = {
        "name": "calculate_sum",
        "description": "Calculates the sum of two integers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }
    }

    skills = importer.import_skills(skill_def)
    assert len(skills) == 1
    skill_func = skills[0]
    assert skill_func.__name__ == "calculate_sum"
    assert skill_func.__doc__ == "Calculates the sum of two integers."
    assert getattr(skill_func, "__mcp_schema__") == skill_def["parameters"]

    result = skill_func(a=5, b=10)
    assert "Executed imported skill 'calculate_sum' successfully" in result

def test_importer_list_of_skills() -> None:
    """Tests importing a list of skill dictionaries."""
    importer = AgentSkillsImporter()
    skills_list: List[Dict[str, Any]] = [
        {
            "name": "greet",
            "description": "Greets the user by name.",
            "inputSchema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }
        },
        {
            "name": "ping",
            "description": "Returns pong."
        }
    ]

    skills = importer.import_skills(skills_list)
    assert len(skills) == 2
    assert skills[0].__name__ == "greet"
    assert skills[0].__doc__ == "Greets the user by name."
    assert getattr(skills[0], "__mcp_schema__") == skills_list[0]["inputSchema"]

    assert skills[1].__name__ == "ping"
    assert skills[1].__doc__ == "Returns pong."
    assert not hasattr(skills[1], "__mcp_schema__")

def test_importer_wrapped_skills_json() -> None:
    """Tests importing skills from a JSON string representing a wrapped skills object."""
    importer = AgentSkillsImporter()
    wrapped_data = {
        "version": "1.0",
        "skills": [
            {
                "name": "search",
                "description": "Search the web."
            }
        ]
    }
    json_str = json.dumps(wrapped_data)

    skills = importer.import_skills(json_str)
    assert len(skills) == 1
    assert skills[0].__name__ == "search"
    assert skills[0].__doc__ == "Search the web."

def test_importer_with_registry() -> None:
    """Tests importing skills and registering them automatically into SkillRegistry."""
    registry = SkillRegistry()
    importer = AgentSkillsImporter(registry=registry)

    skill_def: Dict[str, Any] = {
        "name": "fetch_data",
        "description": "Fetches remote data.",
        "parameters": {"type": "object", "properties": {}}
    }

    skills = importer.import_skills(skill_def)
    assert len(skills) == 1

    assert registry.has_skill("fetch_data")
    assert registry.descriptions["fetch_data"] == "Fetches remote data."

    # Verify execution via registry
    result = registry.execute_skill("fetch_data")
    assert "Executed imported skill 'fetch_data' successfully" in result

def test_importer_invalid_schemas() -> None:
    """Tests that the importer gracefully handles invalid schemas."""
    importer = AgentSkillsImporter()

    # Missing name
    invalid_1 = {"description": "A skill without name"}
    assert not importer.validate_skill_definition(invalid_1)

    # Empty name
    invalid_2 = {"name": "", "description": "A skill with empty name"}
    assert not importer.validate_skill_definition(invalid_2)

    # Missing description
    invalid_3 = {"name": "test"}
    assert not importer.validate_skill_definition(invalid_3)

    # Not a dict
    assert not importer.validate_skill_definition([])  # type: ignore

    # Import list with one valid and one invalid
    mixed_list = [
        {"name": "valid", "description": "Valid skill"},
        {"name": "invalid"}  # missing description
    ]

    imported = importer.import_skills(mixed_list)
    assert len(imported) == 1
    assert imported[0].__name__ == "valid"

def test_importer_invalid_json_string() -> None:
    """Tests importing from an invalid JSON string."""
    importer = AgentSkillsImporter()
    imported = importer.import_skills("invalid json strings here{")
    assert len(imported) == 0

def test_importer_unsupported_format() -> None:
    """Tests importing from unsupported format type like integers."""
    importer = AgentSkillsImporter()
    imported = importer.import_skills(12345)  # type: ignore
    assert len(imported) == 0
