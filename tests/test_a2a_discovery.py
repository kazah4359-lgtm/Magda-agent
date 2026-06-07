import pytest
import json
from magda_agent.integration.a2a_discovery import AgentCard, A2ADiscovery

@pytest.fixture
def local_card():
    return AgentCard(
        agent_id="agent-001",
        name="MagdaLocal",
        description="Local agent for testing",
        capabilities=["chat", "code_execution"],
        endpoints={"rpc": "http://localhost:8080/rpc"}
    )

@pytest.fixture
def remote_card_1():
    return AgentCard(
        agent_id="agent-remote-001",
        name="RemoteWorker1",
        description="Worker node",
        capabilities=["image_generation", "chat"],
        endpoints={"rpc": "http://192.168.1.10:8080/rpc"}
    )

@pytest.fixture
def remote_card_2():
    return AgentCard(
        agent_id="agent-remote-002",
        name="RemoteWorker2",
        description="Code worker node",
        capabilities=["code_execution", "linting"],
        endpoints={"rpc": "http://192.168.1.11:8080/rpc"}
    )

@pytest.mark.asyncio
async def test_broadcast_card(local_card):
    discovery = A2ADiscovery(local_card=local_card)
    broadcasted_json = await discovery.broadcast_card()

    data = json.loads(broadcasted_json)
    assert data["agent_id"] == "agent-001"
    assert data["name"] == "MagdaLocal"
    assert "chat" in data["capabilities"]

@pytest.mark.asyncio
async def test_fetch_and_index_cards(local_card, remote_card_1, remote_card_2):
    discovery = A2ADiscovery(local_card=local_card)

    mock_network_cards = [
        remote_card_1.to_json(),
        remote_card_2.to_json()
    ]

    await discovery.fetch_cards(mock_network_cards=mock_network_cards)

    # Test getting by id
    fetched_agent = discovery.get_agent_by_id("agent-remote-001")
    assert fetched_agent is not None
    assert fetched_agent.name == "RemoteWorker1"

    # Test indexing by capability (chat)
    chat_agents = discovery.find_agents_by_capability("chat")
    assert len(chat_agents) == 1
    assert chat_agents[0].agent_id == "agent-remote-001"

    # Test indexing by capability (code_execution)
    code_agents = discovery.find_agents_by_capability("code_execution")
    assert len(code_agents) == 1
    assert code_agents[0].agent_id == "agent-remote-002"

    # Test missing capability
    missing_agents = discovery.find_agents_by_capability("unknown_cap")
    assert len(missing_agents) == 0

@pytest.mark.asyncio
async def test_fetch_invalid_card_json(local_card):
    discovery = A2ADiscovery(local_card=local_card)

    # Passing invalid JSON should be caught and not crash
    mock_network_cards = [
        '{"invalid": "json"',
        '{"agent_id": "missing_fields"}' # Missing required fields
    ]

    await discovery.fetch_cards(mock_network_cards=mock_network_cards)

    # No agents should be discovered
    assert len(discovery._discovered_agents) == 0
