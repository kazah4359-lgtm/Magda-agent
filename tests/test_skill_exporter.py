import pytest
from unittest.mock import MagicMock
from magda_agent.skills.exporter import SkillExporter
from magda_agent.memory.procedural import ProceduralMemory
import json
import yaml

def test_export_skill_json():
    mock_memory = MagicMock(spec=ProceduralMemory)
    mock_memory.get_procedure_versions.return_value = {
        "documents": ["Procedure Name: test_skill\nProcedure: def my_func():\n    return 'hello'"],
        "metadatas": [{"name": "test_skill", "user_id": 1}]
    }

    exporter = SkillExporter(mock_memory)
    result_json = exporter.export_skill("test_skill", export_format="json", user_id=1)

    data = json.loads(result_json)
    assert data["metadata"]["name"] == "test_skill"
    assert data["metadata"]["user_id"] == 1
    assert data["code"] == "def my_func():\n    return 'hello'"

def test_export_skill_yaml():
    mock_memory = MagicMock(spec=ProceduralMemory)
    mock_memory.get_procedure_versions.return_value = {
        "documents": ["Procedure Name: test_skill\nProcedure: def my_func():\n    return 'hello'"],
        "metadatas": [{"name": "test_skill", "user_id": 1}]
    }

    exporter = SkillExporter(mock_memory)
    result_yaml = exporter.export_skill("test_skill", export_format="yaml", user_id=1)

    data = yaml.safe_load(result_yaml)
    assert data["metadata"]["name"] == "test_skill"
    assert data["metadata"]["user_id"] == 1
    assert data["code"] == "def my_func():\n    return 'hello'"

def test_export_skill_not_found():
    mock_memory = MagicMock(spec=ProceduralMemory)
    mock_memory.get_procedure_versions.return_value = {}

    exporter = SkillExporter(mock_memory)
    with pytest.raises(ValueError, match="not found in procedural memory"):
        exporter.export_skill("missing_skill")

def test_export_skill_invalid_format():
    mock_memory = MagicMock(spec=ProceduralMemory)
    mock_memory.get_procedure_versions.return_value = {
        "documents": ["Procedure Name: test_skill\nProcedure: def my_func():\n    return 'hello'"],
        "metadatas": [{"name": "test_skill"}]
    }

    exporter = SkillExporter(mock_memory)
    with pytest.raises(ValueError, match="Unsupported format"):
        exporter.export_skill("test_skill", export_format="xml")
