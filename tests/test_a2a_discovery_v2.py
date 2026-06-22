import pytest
import json
from magda_agent.integration.a2a_discovery_v2 import AgentCardV2, A2ADiscoveryV2

@pytest.fixture
def local_card_v2() -> AgentCardV2:
    return AgentCardV2(
        agent_id="agent-v2-001",
        name="MagdaLocalV2",
        description="Local agent for testing v2",
        capabilities=["chat", "code_execution"],
        endpoints={"rpc": "http://localhost:8080/rpc"}
    )

@pytest.fixture
def remote_card_1_v2() -> AgentCardV2:
    return AgentCardV2(
        agent_id="agent-remote-v2-001",
        name="RemoteWorker1V2",
        description="Worker node v2",
        capabilities=["image_generation", "chat"],
        endpoints={"rpc": "http://192.168.1.10:8080/rpc"}
    )

@pytest.fixture
def remote_card_2_v2() -> AgentCardV2:
    return AgentCardV2(
        agent_id="agent-remote-v2-002",
        name="RemoteWorker2V2",
        description="Code worker node v2",
        capabilities=["code_execution", "linting"],
        endpoints={"rpc": "http://192.168.1.11:8080/rpc"}
    )

@pytest.mark.asyncio
async def test_broadcast_card_v2(local_card_v2: AgentCardV2) -> None:
    discovery = A2ADiscoveryV2(local_card=local_card_v2)
    broadcasted_json = await discovery.broadcast_card()

    envelope = json.loads(broadcasted_json)
    assert envelope["type"] == "a2a_discovery_broadcast"
    assert envelope["version"] == "2.0"

    payload = json.loads(envelope["payload"])
    assert payload["agent_id"] == "agent-v2-001"
    assert payload["name"] == "MagdaLocalV2"
    assert "chat" in payload["capabilities"]
    assert payload["protocol_version"] == "v2"

@pytest.mark.asyncio
async def test_fetch_and_index_cards_v2(local_card_v2: AgentCardV2, remote_card_1_v2: AgentCardV2, remote_card_2_v2: AgentCardV2) -> None:
    discovery = A2ADiscoveryV2(local_card=local_card_v2)

    network_envelopes = [
        json.dumps({"type": "a2a_discovery_broadcast", "version": "2.0", "payload": remote_card_1_v2.to_json()}),
        json.dumps({"type": "a2a_discovery_broadcast", "version": "2.0", "payload": remote_card_2_v2.to_json()})
    ]

    await discovery.fetch_cards(network_envelopes=network_envelopes)

    # Test getting by id
    fetched_agent = discovery.get_agent_by_id("agent-remote-v2-001")
    assert fetched_agent is not None
    assert fetched_agent.name == "RemoteWorker1V2"

    # Test indexing by capability (chat)
    chat_agents = discovery.find_agents_by_capability("chat")
    assert len(chat_agents) == 1
    assert chat_agents[0].agent_id == "agent-remote-v2-001"

    # Test indexing by capability (code_execution)
    code_agents = discovery.find_agents_by_capability("code_execution")
    assert len(code_agents) == 1
    assert code_agents[0].agent_id == "agent-remote-v2-002"

    # Test missing capability
    missing_agents = discovery.find_agents_by_capability("unknown_cap")
    assert len(missing_agents) == 0

@pytest.mark.asyncio
async def test_fetch_invalid_card_json_v2(local_card_v2: AgentCardV2) -> None:
    discovery = A2ADiscoveryV2(local_card=local_card_v2)

    # Passing invalid JSON should be caught and not crash
    network_envelopes = [
        '{"invalid": "json"',
        json.dumps({"type": "wrong_type", "version": "2.0", "payload": "{}"}),
        json.dumps({"type": "a2a_discovery_broadcast", "version": "1.0", "payload": "{}"}),
        json.dumps({"type": "a2a_discovery_broadcast", "version": "2.0", "payload": '{"agent_id": "missing_fields"}'})
    ]

    await discovery.fetch_cards(network_envelopes=network_envelopes)

    # No agents should be discovered
    assert len(discovery._discovered_agents) == 0
