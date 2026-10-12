import pytest
from typing import List, Dict, Any
from magda_agent.skills.mcp_export_v8 import MCPDynamicExporterV8
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.mcp_registry import MCPRegistry


def sample_skill_v8(a: int, b: float, c: bool, d: List[str], e: Dict[str, Any] = None, f: str = "default") -> str:
    """Sample skill for V8 test."""
    return f"{a} - {b} - {c} - {d} - {e} - {f}"


def test_export_skill_to_mcp_v8_success():
    """Test successfully exporting a single skill to MCP in V8."""
    skill_registry = SkillRegistry()
    skill_registry.register_skill(
        name="sample_skill_v8",
        func=sample_skill_v8,
        description="A sample test skill for V8."
    )

    mcp_registry = MCPRegistry()
    exporter = MCPDynamicExporterV8(skill_registry, mcp_registry)

    success = exporter.export_skill_to_mcp("sample_skill_v8")

    assert success is True
    assert "sample_skill_v8" in mcp_registry.mcp_tools

    tool = mcp_registry.get_tool("sample_skill_v8")
    assert tool["name"] == "sample_skill_v8"
    assert tool["description"] == "A sample test skill for V8."
    assert tool["inputSchema"]["type"] == "object"

    props = tool["inputSchema"]["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "number"
    assert props["c"]["type"] == "boolean"
    assert props["d"]["type"] == "array"
    assert props["f"]["type"] == "string"

    reqs = tool["inputSchema"]["required"]
    assert "a" in reqs
    assert "b" in reqs
    assert "c" in reqs
    assert "d" in reqs
    assert "f" not in reqs


def test_export_skill_v8_custom_mcp_schema():
    """Test exporting a skill with pre-defined __mcp_schema__."""
    def custom_schema_skill(x):
        return x

    custom_schema_skill.__mcp_schema__ = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"]
    }

    skill_registry = SkillRegistry()
    skill_registry.register_skill(
        name="custom_schema_skill",
        func=custom_schema_skill,
        description="Skill with custom schema"
    )

    mcp_registry = MCPRegistry()
    exporter = MCPDynamicExporterV8(skill_registry, mcp_registry)

    success = exporter.export_skill_to_mcp("custom_schema_skill")

    assert success is True
    tool = mcp_registry.get_tool("custom_schema_skill")
    assert tool["inputSchema"] == custom_schema_skill.__mcp_schema__


def test_export_skill_v8_not_found():
    """Test exporting a non-existent skill in V8."""
    skill_registry = SkillRegistry()
    mcp_registry = MCPRegistry()
    exporter = MCPDynamicExporterV8(skill_registry, mcp_registry)

    success = exporter.export_skill_to_mcp("non_existent_skill_v8")

    assert success is False
    assert len(mcp_registry.mcp_tools) == 0


def test_export_all_skills_v8():
    """Test exporting all registered skills in V8."""
    skill_registry = SkillRegistry()
    skill_registry.register_skill(name="skill_1", func=lambda x: x, description="Skill 1")
    skill_registry.register_skill(name="skill_2", func=lambda y: y, description="Skill 2")

    mcp_registry = MCPRegistry()
    exporter = MCPDynamicExporterV8(skill_registry, mcp_registry)

    exported_count = exporter.export_all_skills()

    assert exported_count == 2
    assert "skill_1" in mcp_registry.mcp_tools
    assert "skill_2" in mcp_registry.mcp_tools
