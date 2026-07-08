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


from magda_agent.memory.marketplace import export_episodic_to_marketplace_format, import_episodic_from_marketplace_format

def test_export_episodic_to_marketplace_format():
    events = [
        {
            "id": "mem-1",
            "text": "User wants a pizza",
            "metadata": {"timestamp": 1234567890, "user_id": 42}
        },
        {
            "id": "mem-2",
            "text": "User likes mushrooms",
            "metadata": {"user_id": 42}
        }
    ]

    result = export_episodic_to_marketplace_format(events, author_id="magda_test")

    assert result["version"] == "1.0"
    assert result["author_id"] == "magda_test"
    assert len(result["events"]) == 2

    assert result["events"][0]["event_id"] == "mem-1"
    assert result["events"][0]["text"] == "User wants a pizza"
    assert result["events"][0]["metadata"] == {"timestamp": 1234567890, "user_id": 42}
    assert result["events"][0]["timestamp"] == 1234567890

    assert result["events"][1]["event_id"] == "mem-2"
    assert result["events"][1]["text"] == "User likes mushrooms"
    assert result["events"][1]["metadata"] == {"user_id": 42}
    assert result["events"][1]["timestamp"] is None

def test_import_episodic_from_marketplace_format():
    marketplace_data = {
        "version": "1.0",
        "author_id": "other_agent",
        "events": [
            {
                "event_id": "ext-1",
                "text": "User hates olives",
                "metadata": {"source": "external"},
                "timestamp": 987654321
            },
            {
                "event_id": "ext-2",
                "text": "User loves cheese",
                "metadata": {}
            }
        ]
    }

    result = import_episodic_from_marketplace_format(marketplace_data)

    assert len(result) == 2

    assert result[0]["id"] == "ext-1"
    assert result[0]["text"] == "User hates olives"
    assert result[0]["metadata"]["source"] == "external"
    assert result[0]["metadata"]["timestamp"] == 987654321

    assert result[1]["id"] == "ext-2"
    assert result[1]["text"] == "User loves cheese"
    assert "timestamp" not in result[1]["metadata"]
