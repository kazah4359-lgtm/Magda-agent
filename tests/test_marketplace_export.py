import pytest
import json
from unittest.mock import MagicMock
from magda_agent.skills.marketplace_export import HermesMarketplaceExporterV3

def test_extract_python_code():
    exporter = HermesMarketplaceExporterV3()
    doc = "Procedure Name: test_skill\nProcedure: def my_skill(x: int):\n    return x * 2"
    code = exporter._extract_python_code(doc)
    assert code == "def my_skill(x: int):\n    return x * 2"

    # Test robust extraction with leading spaces/newlines
    doc2 = "Random header\nProcedure:\n   def another_skill(y: str): pass"
    code2 = exporter._extract_python_code(doc2)
    assert code2 == "def another_skill(y: str): pass"

    # Test raw code fallback
    doc_raw = "def raw_skill(): pass"
    assert exporter._extract_python_code(doc_raw) == "def raw_skill(): pass"

    doc_invalid = "No procedure here"
    assert exporter._extract_python_code(doc_invalid) is None

def test_parse_code_to_schema_complex():
    exporter = HermesMarketplaceExporterV3()
    code = """def complex_skill(items: List[int], config: Optional[Dict] = None):
    \"\"\"A more complex skill.\"\"\"
    pass
"""
    schema = exporter._parse_code_to_schema(code)
    assert schema["name"] == "complex_skill"
    assert schema["parameters"]["properties"]["items"]["type"] == "array"
    assert schema["parameters"]["properties"]["items"]["items"] == {"type": "integer"}
    assert schema["parameters"]["properties"]["config"]["type"] == "object"
    assert "items" in schema["parameters"]["required"]
    assert "config" not in schema["parameters"]["required"]

def test_export_procedural_skills():
    exporter = HermesMarketplaceExporterV3()
    memory_mock = MagicMock()

    # Mock documents in the format ProceduralMemory.store_procedure uses
    documents = [
        "Procedure Name: skill1\nProcedure: def skill1(a: int, b: str = 'default'):\n    \"\"\"Doc1\"\"\"\n    pass",
        "Procedure Name: skill2\nProcedure: def skill2(flag: bool):\n    \"\"\"Doc2\"\"\"\n    pass",
        "Not a python skill"
    ]
    metadatas = [
        {"type": "hermes_experience_skill", "user_id": 123},
        {"type": "python_code", "user_id": 123},
        {"type": "text", "user_id": 123}
    ]

    memory_mock.collection.get.return_value = {
        "documents": documents,
        "metadatas": metadatas
    }

    result = exporter.export_procedural_skills(memory_mock, user_id=123)

    assert len(result["skills"]) == 2

    s1 = next(s for s in result["skills"] if s["name"] == "skill1")
    assert s1["description"] == "Doc1"
    assert s1["parameters"]["properties"]["a"]["type"] == "integer"
    assert s1["parameters"]["properties"]["b"]["type"] == "string"
    assert "a" in s1["parameters"]["required"]
    assert "b" not in s1["parameters"]["required"]

    s2 = next(s for s in result["skills"] if s["name"] == "skill2")
    assert s2["description"] == "Doc2"
    assert s2["parameters"]["properties"]["flag"]["type"] == "boolean"
    assert "flag" in s2["parameters"]["required"]

def test_export_to_json():
    exporter = HermesMarketplaceExporterV3()
    memory_mock = MagicMock()
    memory_mock.collection.get.return_value = {
        "documents": ["Procedure Name: s\nProcedure: def s(): pass"],
        "metadatas": [{}]
    }

    json_str = exporter.export_to_json(memory_mock)
    data = json.loads(json_str)
    assert "skills" in data
    assert len(data["skills"]) == 1
    assert data["skills"][0]["name"] == "s"
