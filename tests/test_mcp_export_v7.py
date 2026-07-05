import pytest
from magda_agent.skills.mcp_export_v7 import MCPDynamicExporterV7
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.mcp_registry import MCPRegistry

def sample_skill(a: int, b: str = "default") -> str:
    """Sample skill description."""
    return f"{a} - {b}"

def test_export_skill_to_mcp_success():
    """Test successfully exporting a single skill to MCP."""
    skill_registry = SkillRegistry()
    skill_registry.register_skill(
        name="sample_skill",
        func=sample_skill,
        description="A sample test skill."
    )

    mcp_registry = MCPRegistry()
    exporter = MCPDynamicExporterV7(skill_registry, mcp_registry)

    # Export the skill
    success = exporter.export_skill_to_mcp("sample_skill")

    assert success is True
    assert "sample_skill" in mcp_registry.mcp_tools

    tool = mcp_registry.get_tool("sample_skill")
    assert tool["name"] == "sample_skill"
    assert tool["description"] == "A sample test skill."
    assert tool["inputSchema"]["type"] == "object"
    assert "a" in tool["inputSchema"]["properties"]
    assert "b" in tool["inputSchema"]["properties"]
    assert tool["inputSchema"]["properties"]["a"]["type"] == "integer"
    assert tool["inputSchema"]["properties"]["b"]["type"] == "string"
    assert "a" in tool["inputSchema"]["required"]
    assert "b" not in tool["inputSchema"]["required"]

def test_export_skill_not_found():
    """Test behavior when trying to export a non-existent skill."""
    skill_registry = SkillRegistry()
    mcp_registry = MCPRegistry()
    exporter = MCPDynamicExporterV7(skill_registry, mcp_registry)

    success = exporter.export_skill_to_mcp("non_existent_skill")

    assert success is False
    assert len(mcp_registry.mcp_tools) == 0

def test_export_all_skills():
    """Test exporting multiple skills at once."""
    skill_registry = SkillRegistry()
    skill_registry.register_skill(name="skill_1", func=lambda x: x, description="Skill 1")
    skill_registry.register_skill(name="skill_2", func=lambda y: y, description="Skill 2")

    mcp_registry = MCPRegistry()
    exporter = MCPDynamicExporterV7(skill_registry, mcp_registry)

    exported_count = exporter.export_all_skills()

    assert exported_count == 2
    assert "skill_1" in mcp_registry.mcp_tools
    assert "skill_2" in mcp_registry.mcp_tools
