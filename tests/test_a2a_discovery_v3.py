import pytest
import json
from dataclasses import asdict
from magda_agent.integration.a2a_discovery import AgentCard
from magda_agent.integration.a2a_discovery_v3 import A2ADiscoveryV3

@pytest.fixture
def valid_card():
    return AgentCard(
        agent_id="test-agent-001",
        name="TestAgent",
        description="A test agent",
        capabilities=["test", "parse"],
        endpoints={"rpc": "http://localhost:8080"}
    )

def test_parse_and_register_valid_cards(valid_card):
    discovery = A2ADiscoveryV3()

    valid_json = valid_card.to_json()

    parsed_cards = discovery.parse_and_register_cards([valid_json])

    assert len(parsed_cards) == 1
    assert parsed_cards[0].agent_id == "test-agent-001"
    assert parsed_cards[0].name == "TestAgent"

    # Verify it was registered
    registered_card = discovery.get_agent_card("test-agent-001")
    assert registered_card is not None
    assert registered_card.name == "TestAgent"

    all_agents = discovery.get_all_agents()
    assert len(all_agents) == 1

def test_parse_invalid_json():
    discovery = A2ADiscoveryV3()

    invalid_json = '{"agent_id": "missing_quotes}'

    parsed_cards = discovery.parse_and_register_cards([invalid_json])

    assert len(parsed_cards) == 0
    assert len(discovery.get_all_agents()) == 0

def test_parse_missing_required_fields():
    discovery = A2ADiscoveryV3()

    missing_fields_json = '{"agent_id": "test-agent-002", "name": "Agent"}' # missing description, capabilities, endpoints

    parsed_cards = discovery.parse_and_register_cards([missing_fields_json])

    assert len(parsed_cards) == 0
    assert len(discovery.get_all_agents()) == 0

def test_parse_mixed_valid_and_invalid(valid_card):
    discovery = A2ADiscoveryV3()

    valid_json = valid_card.to_json()
    invalid_json = '{"broken": json}'

    parsed_cards = discovery.parse_and_register_cards([invalid_json, valid_json])

    assert len(parsed_cards) == 1
    assert parsed_cards[0].agent_id == "test-agent-001"
    assert len(discovery.get_all_agents()) == 1
