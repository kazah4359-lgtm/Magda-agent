import pytest
import json
import respx
import httpx
from dataclasses import asdict
from magda_agent.integration.a2a_discovery import AgentCard, A2ADiscovery
from magda_agent.agents.a2a_discovery import A2ADiscoveryAgent

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

@pytest.mark.asyncio
@respx.mock
async def test_register_with_registry(local_card):
    discovery = A2ADiscovery(local_card=local_card)
    registry_url = "http://discovery-registry.local"

    # Mock the registration endpoint
    route = respx.post(f"{registry_url}/register").mock(return_value=httpx.Response(201))

    success = await discovery.register_with_registry(registry_url, auth_token="test-token")

    assert success is True
    assert route.called
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-token"

    # Verify the payload
    sent_data = json.loads(route.calls.last.request.content)
    assert sent_data["agent_id"] == local_card.agent_id
    assert sent_data["protocol_version"] == "2.0"

@pytest.mark.asyncio
@respx.mock
async def test_discover_from_registry(local_card, remote_card_1, remote_card_2):
    discovery = A2ADiscovery(local_card=local_card)
    registry_url = "http://discovery-registry.local"

    # Mock the discovery endpoint
    cards_data = [asdict(remote_card_1), asdict(remote_card_2)]
    route = respx.get(f"{registry_url}/cards").mock(return_value=httpx.Response(200, json=cards_data))

    discovered = await discovery.discover_from_registry(registry_url)

    assert len(discovered) == 2
    assert route.called
    assert discovery.get_agent_by_id(remote_card_1.agent_id).name == remote_card_1.name
    assert discovery.get_agent_by_id(remote_card_2.agent_id).name == remote_card_2.name
    assert remote_card_1.name in [c.name for c in discovery.find_agents_by_capability("chat")]

@pytest.mark.asyncio
@respx.mock
async def test_register_with_registry_failure(local_card):
    discovery = A2ADiscovery(local_card=local_card)
    registry_url = "http://discovery-registry.local"

    # Mock a failure
    respx.post(f"{registry_url}/register").mock(return_value=httpx.Response(500))

    success = await discovery.register_with_registry(registry_url)
    assert success is False

# New tests for A2ADiscoveryAgent
@pytest.mark.asyncio
async def test_a2a_discovery_agent_card_generation():
    agent = A2ADiscoveryAgent("magda-007", "James Magda", ["espionage", "chat"])
    card = agent.generate_my_card(endpoints={"rpc": "http://magda.spy:5000/rpc"})

    assert card.agent_id == "magda-007"
    assert card.name == "James Magda"
    assert "espionage" in card.capabilities
    assert card.endpoints["rpc"] == "http://magda.spy:5000/rpc"

@pytest.mark.asyncio
async def test_a2a_discovery_agent_parsing_and_finding():
    agent = A2ADiscoveryAgent("magda-main", "Magda Main", ["coordination"])
    remote_card = AgentCard(
        agent_id="external-001",
        name="LangGraphWorker",
        description="LangGraph-based worker",
        capabilities=["graph_reasoning"],
        endpoints={"rpc": "http://langgraph.worker:8000/rpc"}
    )

    agent.parse_card(remote_card.to_json())
    assert "external-001" in agent.known_agents

    peers = agent.find_peers_by_capability("graph_reasoning")
    assert len(peers) == 1
    assert peers[0].agent_id == "external-001"

@pytest.mark.asyncio
async def test_a2a_discovery_agent_mock_delegation():
    agent = A2ADiscoveryAgent("magda-main", "Magda Main", ["coordination"])
    external_card = AgentCard(
        agent_id="crewai-001",
        name="CrewAIWorker",
        description="CrewAI-based worker",
        capabilities=["multi_agent_collab"],
        endpoints={} # No real endpoint, should trigger mock
    )
    agent.known_agents[external_card.agent_id] = external_card

    task = {"goal": "collaborate", "data": [1, 2, 3]}
    result = await agent.delegate_to_external("crewai-001", task)

    assert result["status"] == "success"
    assert "Mocked delegation to CrewAIWorker" in result["message"]
    assert result["framework"] == "CrewAI"
    assert result["delegated_task"] == task

@pytest.mark.asyncio
@respx.mock
async def test_a2a_discovery_agent_real_delegation():
    agent = A2ADiscoveryAgent("magda-main", "Magda Main", ["coordination"])
    external_card = AgentCard(
        agent_id="langgraph-001",
        name="LangGraphWorker",
        description="LangGraph-based worker",
        capabilities=["graph_reasoning"],
        endpoints={"rpc": "http://langgraph.worker:8000/rpc"}
    )
    agent.known_agents[external_card.agent_id] = external_card

    task = {"goal": "reason", "node": "start"}
    mock_response = {"status": "completed", "path": ["start", "end"]}

    respx.post("http://langgraph.worker:8000/rpc").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    result = await agent.delegate_to_external("langgraph-001", task)

    assert result["status"] == "completed"
    assert result["path"] == ["start", "end"]
