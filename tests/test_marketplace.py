import pytest
from unittest.mock import patch, AsyncMock
from magda_agent.skills.marketplace import (
    _create_dynamic_skill,
    fetch_and_register_skills,
    search_marketplace_skills
)
from magda_agent.skills.registry import SkillRegistry

def test_create_dynamic_skill_parameters():
    skill_def = {
        "name": "test_skill",
        "description": "A test skill",
        "parameters": {
            "type": "object",
            "properties": {"arg1": {"type": "string"}}
        }
    }
    func = _create_dynamic_skill(skill_def)
    assert func.__name__ == "test_skill"
    assert func.__doc__ == "A test skill"
    assert hasattr(func, "__mcp_schema__")
    assert getattr(func, "__mcp_schema__") == skill_def["parameters"]

def test_create_dynamic_skill_input_schema():
    skill_def = {
        "name": "mcp_tool",
        "description": "An MCP tool",
        "inputSchema": {
            "type": "object",
            "properties": {"arg1": {"type": "string"}}
        }
    }
    func = _create_dynamic_skill(skill_def)
    assert func.__name__ == "mcp_tool"
    assert func.__doc__ == "An MCP tool"
    assert hasattr(func, "__mcp_schema__")
    assert getattr(func, "__mcp_schema__") == skill_def["inputSchema"]

@pytest.mark.asyncio
@patch('aiohttp.ClientSession.get')
async def test_fetch_and_register_skills(mock_get):
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.raise_for_status = lambda: None
    mock_response.json.return_value = {
        "skills": [
            {
                "name": "remote_skill_1",
                "description": "desc 1",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "remote_skill_2",
                "description": "desc 2",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]
    }
    mock_response.__aenter__.return_value = mock_response
    mock_get.return_value = mock_response

    registry = SkillRegistry()
    registered_skills = await fetch_and_register_skills("http://test.url/api", registry)

    assert len(registered_skills) == 2
    assert "remote_skill_1" in registered_skills
    assert "remote_skill_2" in registered_skills

    assert registry.has_skill("remote_skill_1")
    assert registry.has_skill("remote_skill_2")

    skill1 = registry.skills["remote_skill_1"]
    assert getattr(skill1, "__mcp_schema__") == {"type": "object", "properties": {}}

    skill2 = registry.skills["remote_skill_2"]
    assert getattr(skill2, "__mcp_schema__") == {"type": "object", "properties": {}}

@pytest.mark.asyncio
@patch('aiohttp.ClientSession.get')
async def test_search_marketplace_skills(mock_get):
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.raise_for_status = lambda: None
    mock_response.json.return_value = {
        "skills": [
            {
                "name": "calculator",
                "description": "Math operations"
            },
            {
                "name": "weather",
                "description": "Get current weather"
            },
            {
                "name": "math_helper",
                "description": "Advanced calculations"
            }
        ]
    }
    mock_response.__aenter__.return_value = mock_response
    mock_get.return_value = mock_response

    # Search by name
    results = await search_marketplace_skills("http://test.url/api", "weather")
    assert len(results) == 1
    assert results[0]["name"] == "weather"

    # Search by description substring
    results2 = await search_marketplace_skills("http://test.url/api", "math")
    assert len(results2) == 2
    names = [r["name"] for r in results2]
    assert "calculator" in names
    assert "math_helper" in names
