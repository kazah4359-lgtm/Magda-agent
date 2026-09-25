import pytest
import json
import jsonschema
from unittest.mock import patch, AsyncMock
from magda_agent.integration.a2a_discovery_v5 import AgentCardV5, A2ADiscoveryServiceV5

@pytest.fixture
def valid_card_dict() -> dict:
    """Fixture for valid AgentCard dictionary."""
    return {
        "agent_id": "test-agent-005",
        "name": "TestAgentV5",
        "description": "A test agent card v5",
        "capabilities": ["test_capability_1", "test_capability_2"],
        "endpoints": {"rpc": "http://localhost:8080/rpc"},
        "protocol_version": "v5",
        "health_status": "online"
    }

@pytest.fixture
def valid_card_json(valid_card_dict: dict) -> str:
    """Fixture for valid AgentCard JSON string."""
    return json.dumps(valid_card_dict)

def test_agent_card_v5_serialization(valid_card_dict: dict, valid_card_json: str) -> None:
    """
    Test that AgentCardV5 successfully serializes and deserializes from JSON.
    """
    # Deserialize
    card = AgentCardV5.from_json(valid_card_json)
    assert card.agent_id == "test-agent-005"
    assert card.name == "TestAgentV5"
    assert "test_capability_1" in card.capabilities
    assert card.endpoints["rpc"] == "http://localhost:8080/rpc"
    assert card.protocol_version == "v5"

    # Serialize
    re_serialized = card.to_json()
    re_dict = json.loads(re_serialized)
    assert re_dict["agent_id"] == "test-agent-005"
    assert re_dict["name"] == "TestAgentV5"
    assert re_dict["protocol_version"] == "v5"
    assert re_dict["health_status"] == "online"

def test_agent_card_v5_schema_validation_type_mismatch(valid_card_dict: dict) -> None:
    """
    Test that AgentCardV5.from_json raises ValidationError on schema/type mismatches.
    """
    bad_card_dict = valid_card_dict.copy()
    bad_card_dict["capabilities"] = "not-a-list"

    with pytest.raises(jsonschema.ValidationError):
        AgentCardV5.from_json(json.dumps(bad_card_dict))

    bad_endpoints_dict = valid_card_dict.copy()
    bad_endpoints_dict["endpoints"] = "not-an-object"

    with pytest.raises(jsonschema.ValidationError):
        AgentCardV5.from_json(json.dumps(bad_endpoints_dict))

def test_service_filters_offline_agents(valid_card_dict: dict) -> None:
    """
    Test that the service filters out offline agents correctly.
    """
    service = A2ADiscoveryServiceV5()

    online_card = AgentCardV5.from_json(json.dumps(valid_card_dict))
    service.register_agent(online_card)

    offline_dict = valid_card_dict.copy()
    offline_dict["agent_id"] = "offline-agent-001"
    offline_dict["health_status"] = "offline"
    offline_card = AgentCardV5.from_json(json.dumps(offline_dict))
    service.register_agent(offline_card)

    all_agents = service.get_all_agents()
    assert len(all_agents) == 1
    assert all_agents[0].agent_id == "test-agent-005"

    retrieved = service.get_agent_card("offline-agent-001")
    assert retrieved is not None
    assert retrieved.health_status == "offline"

def test_service_register_and_get(valid_card_json: str) -> None:
    """
    Test registering and retrieving an agent card from the service.
    """
    service = A2ADiscoveryServiceV5()
    card = AgentCardV5.from_json(valid_card_json)

    service.register_agent(card)

    retrieved_card = service.get_agent_card("test-agent-005")
    assert retrieved_card is not None
    assert retrieved_card.name == "TestAgentV5"

    all_agents = service.get_all_agents()
    assert len(all_agents) == 1
    assert all_agents[0].agent_id == "test-agent-005"

def test_discovery_service_validate_card(valid_card_dict: dict) -> None:
    """
    Test A2ADiscoveryServiceV5.validate_card with valid and invalid card dicts.
    """
    service = A2ADiscoveryServiceV5()
    assert service.validate_card(valid_card_dict) is True

    invalid_dict = valid_card_dict.copy()
    del invalid_dict["capabilities"]
    assert service.validate_card(invalid_dict) is False

def test_discovery_service_register_invalid_card_raises() -> None:
    """
    Test that register_agent raises ValueError if card fails schema validation.
    """
    service = A2ADiscoveryServiceV5()
    invalid_card = AgentCardV5(
        agent_id="",
        name="",
        description="test",
        capabilities=["test"],
        endpoints={"rpc": "http://localhost"}
    )
    with pytest.raises(ValueError, match="failed schema validation"):
        service.register_agent(invalid_card)

def test_discovery_service_custom_schema(valid_card_dict: dict) -> None:
    """
    Test A2ADiscoveryServiceV5 with a custom schema constraint.
    """
    custom_schema = {
        "type": "object",
        "properties": {
            "agent_id": {"type": "string"},
            "custom_field": {"type": "string"}
        },
        "required": ["agent_id", "custom_field"]
    }
    service = A2ADiscoveryServiceV5(schema=custom_schema)
    assert service.validate_card({"agent_id": "a1"}) is False
    assert service.validate_card({"agent_id": "a1", "custom_field": "val1"}) is True

def test_service_unregister(valid_card_json: str) -> None:
    """
    Test unregistering an agent card from the service.
    """
    service = A2ADiscoveryServiceV5()
    card = AgentCardV5.from_json(valid_card_json)

    service.register_agent(card)
    assert service.get_agent_card("test-agent-005") is not None

    service.unregister_agent("test-agent-005")
    assert service.get_agent_card("test-agent-005") is None

    # Unregistering non-existent agent should not crash
    service.unregister_agent("non-existent")

def test_service_parse_and_register_cards(valid_card_json: str) -> None:
    """
    Test bulk parsing and registration of valid and invalid card strings.
    """
    service = A2ADiscoveryServiceV5()

    invalid_card_json = '{"agent_id": "bad", "name": "bad"}' # missing description, capabilities, endpoints
    malformed_json = '{"agent_id": "bad", "name":' # Syntax error
    type_mismatch_json = json.dumps({"agent_id": "a", "name": "b", "description": "c", "capabilities": 123, "endpoints": {}})

    cards_to_parse = [valid_card_json, invalid_card_json, malformed_json, type_mismatch_json]

    successfully_parsed = service.parse_and_register_cards(cards_to_parse)

    assert len(successfully_parsed) == 1
    assert successfully_parsed[0].agent_id == "test-agent-005"

    assert len(service.get_all_agents()) == 1

@patch('magda_agent.integration.a2a_discovery_v5.logging.error')
def test_service_parse_logging_errors(mock_logging_error) -> None:
    """
    Test that invalid cards log an error during parsing.
    """
    service = A2ADiscoveryServiceV5()
    malformed_json = '{"not": json}'

    service.parse_and_register_cards([malformed_json])

    assert mock_logging_error.called
    assert "Failed to parse AgentCardV5" in mock_logging_error.call_args[0][0]

@pytest.mark.asyncio
async def test_discover_from_network(valid_card_dict: dict) -> None:
    """
    Test discover_from_network fetches and registers cards correctly using respx.
    """
    import respx
    from httpx import Response

    service = A2ADiscoveryServiceV5()
    endpoint = "http://fake-registry.local/cards"

    with respx.mock:
        # Mock successful response
        respx.get(endpoint).mock(return_value=Response(200, json=[valid_card_dict, valid_card_dict]))

        cards = await service.discover_from_network(endpoint)

        assert len(cards) == 2
        assert cards[0].agent_id == "test-agent-005"

        # Check registry
        assert len(service.get_all_agents()) == 1 # duplicate agent_id overwrites
        assert service.get_agent_card("test-agent-005") is not None

@pytest.mark.asyncio
async def test_discover_from_network_invalid_token() -> None:
    """
    Test discover_from_network with invalid auth token.
    """
    service = A2ADiscoveryServiceV5()
    endpoint = "http://fake-registry.local/cards"

    with patch.object(service.security_context, 'validate_token', return_value=False):
        with pytest.raises(ValueError, match="Invalid authentication token"):
            await service.discover_from_network(endpoint, auth_token="bad-token")

@pytest.mark.asyncio
async def test_discover_from_network_http_error() -> None:
    """
    Test discover_from_network handling http errors.
    """
    import respx
    from httpx import Response

    service = A2ADiscoveryServiceV5()
    endpoint = "http://fake-registry.local/cards"

    with respx.mock:
        respx.get(endpoint).mock(return_value=Response(500))

        cards = await service.discover_from_network(endpoint)

        assert cards == []
        assert len(service.get_all_agents()) == 0
