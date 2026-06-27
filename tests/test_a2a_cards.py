import pytest
import json
from magda_agent.integration.a2a_cards import AgentCardV3, A2ADiscoveryV3


def test_agent_card_v3_serialization():
    card = AgentCardV3(
        agent_id="test-agent",
        name="Test Agent",
        description="A test agent",
        capabilities=["code_execution", "web_search"],
        endpoints={"mcp": "http://localhost:8080"}
    )
    json_str = card.to_json()
    new_card = AgentCardV3.from_json(json_str)
    assert new_card.agent_id == card.agent_id
    assert new_card.capabilities == card.capabilities
    assert new_card.protocol_version == "v3"


def test_agent_card_v3_capability_matching():
    card = AgentCardV3(
        agent_id="test-agent",
        name="Test Agent",
        description="A test agent",
        capabilities=["code_execution", "web_search", "image_gen"],
        endpoints={}
    )

    # Exact match
    assert card.has_capability("code_execution") is True
    assert card.has_capability("web_search") is True

    # Prefix match
    assert card.has_capability("code") is True
    assert card.has_capability("image") is True

    # No match
    assert card.has_capability("translation") is False
    assert card.has_capability("image_generation") is False # image_gen vs image_generation

    # Matches any
    assert card.matches_any_capability(["translation", "web_search"]) is True
    assert card.matches_any_capability(["translation", "data_science"]) is False


@pytest.mark.asyncio
async def test_a2a_discovery_v3():
    local_card = AgentCardV3(
        agent_id="local-agent",
        name="Local Agent",
        description="Local agent",
        capabilities=["orchestration"],
        endpoints={}
    )
    discovery = A2ADiscoveryV3(local_card=local_card)

    # Test broadcast
    broadcast_json = await discovery.broadcast_card()
    broadcast_data = json.loads(broadcast_json)
    assert broadcast_data["type"] == "a2a_discovery_broadcast"
    assert broadcast_data["version"] == "3.0"
    assert broadcast_data["payload"]["agent_id"] == "local-agent"

    # Test fetch and register
    peer_card = AgentCardV3(
        agent_id="peer-agent",
        name="Peer Agent",
        description="Peer agent",
        capabilities=["code_execution"],
        endpoints={}
    )
    peer_envelope = {
        "type": "a2a_discovery_broadcast",
        "version": "3.0",
        "payload": json.loads(peer_card.to_json())
    }

    await discovery.fetch_cards(network_envelopes=[json.dumps(peer_envelope)])

    assert discovery.get_agent_by_id("peer-agent") is not None

    # Test search by capability
    matched = discovery.find_agents_by_capability("code_execution")
    assert len(matched) == 1
    assert matched[0].agent_id == "peer-agent"

    # Test search by prefix capability
    matched_prefix = discovery.find_agents_by_capability("code")
    assert len(matched_prefix) == 1
    assert matched_prefix[0].agent_id == "peer-agent"

    # Test no match
    matched_none = discovery.find_agents_by_capability("translation")
    assert len(matched_none) == 0
