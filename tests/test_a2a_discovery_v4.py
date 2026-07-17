import pytest
import json
from unittest.mock import AsyncMock
from magda_agent.integration.a2a import A2AManager
from magda_agent.integration.a2a_discovery import AgentCard

@pytest.fixture
def local_card() -> AgentCard:
    """Fixture for local AgentCard."""
    return AgentCard(
        agent_id="agent-001",
        name="MagdaLocal",
        description="Local agent for testing",
        capabilities=["chat", "planning"],
        endpoints={"rpc": "http://localhost:8080/rpc"}
    )

@pytest.fixture
def remote_card() -> AgentCard:
    """Fixture for remote AgentCard."""
    return AgentCard(
        agent_id="agent-remote-001",
        name="RemoteWorker",
        description="Worker node",
        capabilities=["code_execution", "linting"],
        endpoints={"rpc": "http://192.168.1.10:8080/rpc"}
    )

@pytest.mark.asyncio
async def test_a2a_manager_start(local_card: AgentCard) -> None:
    """Test A2AManager start functionality."""
    manager = A2AManager(local_card=local_card)
    broadcast_json = await manager.start()

    data = json.loads(broadcast_json)
    assert data["agent_id"] == "agent-001"
    assert "planning" in data["capabilities"]

@pytest.mark.asyncio
async def test_a2a_manager_discover_and_delegate(local_card: AgentCard, remote_card: AgentCard) -> None:
    """Test A2AManager discovery and delegation."""
    manager = A2AManager(local_card=local_card)

    mock_network_cards = [remote_card.to_json()]
    await manager.discover_peers(mock_network_cards=mock_network_cards)

    peers = manager.get_known_peers()
    assert len(peers) == 1
    assert peers[0].agent_id == "agent-remote-001"

    # Test delegation to found peer
    manager.delegator.delegate_subplan = AsyncMock(return_value="Delegated to Agent RemoteWorker")
    result = await manager.delegate_task("code_execution", {"code": "print('hello')"})
    assert result == f"Delegated to Agent RemoteWorker"

    # Test delegation to missing capability
    manager.delegator.delegate_subplan = AsyncMock(return_value="No agent found")
    result_missing = await manager.delegate_task("image_generation", {"prompt": "cat"})
    assert result_missing == "No agent found"

import logging
from unittest.mock import patch
from magda_agent.integration.a2a_discovery_v4 import AgentCardV4, A2ADiscoveryRegistryV4

@pytest.fixture
def valid_card_dict() -> dict:
    """Fixture for valid AgentCard dictionary."""
    return {
        "agent_id": "test-agent-001",
        "name": "TestAgent",
        "description": "A test agent card",
        "capabilities": ["test_capability_1", "test_capability_2"],
        "endpoints": {"rpc": "http://localhost:8080/rpc"}
    }

@pytest.fixture
def valid_card_json(valid_card_dict: dict) -> str:
    """Fixture for valid AgentCard JSON string."""
    return json.dumps(valid_card_dict)

def test_agent_card_v4_serialization(valid_card_dict: dict, valid_card_json: str) -> None:
    """
    Test that AgentCardV4 successfully serializes and deserializes from JSON.
    """
    # Deserialize
    card = AgentCardV4.from_json(valid_card_json)
    assert card.agent_id == "test-agent-001"
    assert card.name == "TestAgent"
    assert "test_capability_1" in card.capabilities
    assert card.endpoints["rpc"] == "http://localhost:8080/rpc"

    # Serialize
    re_serialized = card.to_json()
    re_dict = json.loads(re_serialized)
    assert re_dict["agent_id"] == "test-agent-001"
    assert re_dict["name"] == "TestAgent"

def test_registry_register_and_get(valid_card_json: str) -> None:
    """
    Test registering and retrieving an agent card from the registry.
    """
    registry = A2ADiscoveryRegistryV4()
    card = AgentCardV4.from_json(valid_card_json)

    registry.register_agent(card)

    retrieved_card = registry.get_agent_card("test-agent-001")
    assert retrieved_card is not None
    assert retrieved_card.name == "TestAgent"

    all_agents = registry.get_all_agents()
    assert len(all_agents) == 1
    assert all_agents[0].agent_id == "test-agent-001"

def test_registry_unregister(valid_card_json: str) -> None:
    """
    Test unregistering an agent card from the registry.
    """
    registry = A2ADiscoveryRegistryV4()
    card = AgentCardV4.from_json(valid_card_json)

    registry.register_agent(card)
    assert registry.get_agent_card("test-agent-001") is not None

    registry.unregister_agent("test-agent-001")
    assert registry.get_agent_card("test-agent-001") is None

    # Unregistering non-existent agent should not crash
    registry.unregister_agent("non-existent")

def test_registry_parse_and_register_cards(valid_card_json: str) -> None:
    """
    Test bulk parsing and registration of valid and invalid card strings.
    """
    registry = A2ADiscoveryRegistryV4()

    invalid_card_json = '{"agent_id": "bad", "name": "bad"}' # missing description, capabilities, endpoints
    malformed_json = '{"agent_id": "bad", "name":' # Syntax error

    cards_to_parse = [valid_card_json, invalid_card_json, malformed_json]

    successfully_parsed = registry.parse_and_register_cards(cards_to_parse)

    assert len(successfully_parsed) == 1
    assert successfully_parsed[0].agent_id == "test-agent-001"

    assert len(registry.get_all_agents()) == 1

@patch('magda_agent.integration.a2a_discovery_v4.logging.error')
def test_registry_parse_logging_errors(mock_logging_error) -> None:
    """
    Test that invalid cards log an error during parsing.
    """
    registry = A2ADiscoveryRegistryV4()
    malformed_json = '{"not": json}'

    registry.parse_and_register_cards([malformed_json])

    assert mock_logging_error.called
    assert "Failed to parse AgentCardV4" in mock_logging_error.call_args[0][0]
