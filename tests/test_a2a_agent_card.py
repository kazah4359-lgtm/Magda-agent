import json
import pytest
from magda_agent.skills.registry import SkillRegistry
from magda_agent.a2a.agent_card import generate_agent_card, generate_agent_card_json

def mock_skill_func():
    pass

@pytest.fixture
def mock_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register_skill("search", mock_skill_func, "Searches the internet for information.")
    registry.register_skill("weather", mock_skill_func, "Gets current weather for a location.")
    return registry

def test_generate_agent_card(mock_registry: SkillRegistry):
    endpoint = "https://magda.example.com/api/v1/a2a"
    card = generate_agent_card(mock_registry, endpoint)

    assert card["agent_name"] == "Magda"
    assert card["endpoint"] == endpoint
    assert "text" in card["supported_modalities"]
    assert "protocol" in card
    assert card["protocol"] == "A2A"

    capabilities = card["capabilities"]
    assert len(capabilities) == 2

    # Check capabilities
    search_cap = next((cap for cap in capabilities if cap["name"] == "search"), None)
    assert search_cap is not None
    assert search_cap["description"] == "Searches the internet for information."

    weather_cap = next((cap for cap in capabilities if cap["name"] == "weather"), None)
    assert weather_cap is not None
    assert weather_cap["description"] == "Gets current weather for a location."

def test_generate_agent_card_json(mock_registry: SkillRegistry):
    endpoint = "tcp://192.168.1.10:5050"
    card_json = generate_agent_card_json(mock_registry, endpoint)

    # Parse back to dict
    card = json.loads(card_json)

    assert card["agent_name"] == "Magda"
    assert card["endpoint"] == endpoint
    assert card["capabilities"][0]["name"] == "search"
    assert card["capabilities"][1]["name"] == "weather"
    assert card["version"] == "1.0.0"
