import os
import pytest
import shutil
import tempfile
from unittest.mock import MagicMock
from magda_agent.skills.marketplace import load_skill_from_directory, _create_dynamic_skill
from magda_agent.skills.registry import SkillRegistry

@pytest.fixture
def temp_skill_dir():
    # Setup
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Teardown
    shutil.rmtree(tmpdir)

def test_load_skill_from_directory_full(temp_skill_dir):
    skill_name = "weather_skill"
    skill_desc = "Get weather information"
    skill_params = {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    }

    skill_md_content = f"""---
name: {skill_name}
description: {skill_desc}
parameters: {skill_params}
---
Instructions for weather skill.
"""
    with open(os.path.join(temp_skill_dir, "SKILL.md"), "w") as f:
        f.write(skill_md_content)

    scripts_dir = os.path.join(temp_skill_dir, "scripts")
    os.makedirs(scripts_dir)

    script_content = f"""
def {skill_name}(location: str):
    return f"Weather in {{location}} is sunny"
"""
    with open(os.path.join(scripts_dir, "weather.py"), "w") as f:
        f.write(script_content)

    registry = SkillRegistry()
    name = load_skill_from_directory(temp_skill_dir, registry)

    assert name == skill_name
    assert registry.has_skill(skill_name)
    assert registry.descriptions[skill_name] == skill_desc

    skill_func = registry.skills[skill_name]
    assert skill_func(location="London") == "Weather in London is sunny"
    assert hasattr(skill_func, "__mcp_schema__")
    assert getattr(skill_func, "__mcp_schema__") == skill_params

def test_load_skill_from_directory_no_script(temp_skill_dir):
    skill_name = "no_script_skill"
    skill_desc = "A skill without a script"

    skill_md_content = f"""---
name: {skill_name}
description: {skill_desc}
---
Follow these manual instructions.
"""
    with open(os.path.join(temp_skill_dir, "SKILL.md"), "w") as f:
        f.write(skill_md_content)

    registry = SkillRegistry()
    name = load_skill_from_directory(temp_skill_dir, registry)

    assert name == skill_name
    assert registry.has_skill(skill_name)

    skill_func = registry.skills[skill_name]
    result = skill_func()
    assert "Follow these manual instructions." in result

def test_create_dynamic_skill_with_params():
    skill_def = {
        "name": "dynamic_test",
        "description": "Dynamic test desc",
        "parameters": {
            "type": "object",
            "properties": {"arg1": {"type": "string"}}
        }
    }

    func = _create_dynamic_skill(skill_def)
    assert func.__name__ == "dynamic_test"
    assert func.__doc__ == "Dynamic test desc"
    assert hasattr(func, "__mcp_schema__")
    assert getattr(func, "__mcp_schema__") == skill_def["parameters"]
