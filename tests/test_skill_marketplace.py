import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from magda_agent.skills.marketplace import fetch_and_register_skills
from magda_agent.skills.registry import SkillRegistry

@pytest.mark.asyncio
async def test_fetch_and_register_skills():
    mock_url = "https://mock-marketplace.io/skills.json"
    mock_data = {
        "skills": [
            {
                "name": "mock_skill_1",
                "description": "Mock skill 1"
            },
            {
                "name": "mock_skill_2"
            }
        ]
    }

    registry = MagicMock(spec=SkillRegistry)

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status = MagicMock()

        # Setup context manager mock
        mock_get.return_value.__aenter__.return_value = mock_response

        registered_skills = await fetch_and_register_skills(mock_url, registry)

        assert mock_get.called
        assert mock_response.raise_for_status.called

        assert len(registered_skills) == 2
        assert "mock_skill_1" in registered_skills
        assert "mock_skill_2" in registered_skills

        assert registry.register_skill.call_count == 2

        call_args_1 = registry.register_skill.call_args_list[0][1]
        assert call_args_1["name"] == "mock_skill_1"
        assert call_args_1["description"] == "Mock skill 1"
        assert callable(call_args_1["func"])

        call_args_2 = registry.register_skill.call_args_list[1][1]
        assert call_args_2["name"] == "mock_skill_2"
        assert call_args_2["description"] == "Dynamic marketplace skill"
        assert callable(call_args_2["func"])

@pytest.mark.asyncio
async def test_search_marketplace_skills():
    from magda_agent.skills.marketplace import search_marketplace_skills
    mock_url = "https://mock-marketplace.io/skills.json"
    mock_data = {
        "skills": [
            {
                "name": "data_analysis",
                "description": "A skill to analyze data"
            },
            {
                "name": "text_summarizer",
                "description": "Summarizes large chunks of text"
            },
            {
                "name": "image_generator",
                "description": "Generates images from text"
            }
        ]
    }

    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value.__aenter__.return_value = mock_response

        # Search by name
        results = await search_marketplace_skills(mock_url, "text")
        assert len(results) == 2
        assert any(r["name"] == "text_summarizer" for r in results)
        assert any(r["name"] == "image_generator" for r in results)

        # Search by description
        results = await search_marketplace_skills(mock_url, "analyze")
        assert len(results) == 1
        assert results[0]["name"] == "data_analysis"

        # Search with no match
        results = await search_marketplace_skills(mock_url, "nonexistent")
        assert len(results) == 0
