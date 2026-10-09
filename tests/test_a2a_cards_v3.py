import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import httpx

from magda_agent.integration.a2a_cards_v3 import (
    AgentCardV3,
    A2ADiscoveryCardsV3,
    A2AAgentDiscoveryV3,
)


def test_agent_card_v3_creation_and_serialization():
    card = AgentCardV3(
        agent_id="agent-001",
        name="Test Agent One",
        description="A test agent instance",
        capabilities=["code_execution", "web_search"],
        endpoints={"http": "http://localhost:8000/a2a"},
        metadata={"region": "us-west-2"},
        status="active",
    )

    assert card.agent_id == "agent-001"
    assert card.protocol_version == "v3"

    card_dict = card.to_dict()
    assert card_dict["agent_id"] == "agent-001"
    assert card_dict["capabilities"] == ["code_execution", "web_search"]

    json_str = card.to_json()
    new_card = AgentCardV3.from_json(json_str)
    assert new_card.agent_id == card.agent_id
    assert new_card.capabilities == card.capabilities
    assert new_card.metadata == {"region": "us-west-2"}

    # Test from_dict with missing optional fields
    partial_dict = {
        "agent_id": "agent-002",
        "capabilities": ["data_analysis"],
        "endpoints": {"mcp": "http://localhost:8080"},
    }
    restored = AgentCardV3.from_dict(partial_dict)
    assert restored.agent_id == "agent-002"
    assert restored.name == "agent-002"
    assert restored.protocol_version == "v3"
    assert restored.status == "active"


def test_agent_card_v3_validation_errors():
    with pytest.raises(ValueError, match="non-empty string 'agent_id'"):
        AgentCardV3(
            agent_id="",
            name="Empty ID",
            description="desc",
            capabilities=[],
            endpoints={},
        )

    with pytest.raises(ValueError, match="Missing required field 'capabilities'"):
        AgentCardV3.from_dict({"agent_id": "agent-1", "endpoints": {}})

    with pytest.raises(ValueError, match="Invalid JSON string"):
        AgentCardV3.from_json("not a valid json")


def test_agent_card_v3_capability_matching():
    card = AgentCardV3(
        agent_id="code-bot",
        name="Code Bot",
        description="Handles coding tasks",
        capabilities=["code_execution", "web.search", "data_processing"],
        endpoints={},
    )

    # Exact matches
    assert card.has_capability("code_execution") is True
    assert card.has_capability("web.search") is True

    # Prefix matches
    assert card.has_capability("code") is True
    assert card.has_capability("web") is True
    assert card.has_capability("data") is True

    # Non-matches
    assert card.has_capability("translation") is False
    assert card.has_capability("code_refactoring") is False

    # Matches any / matches all
    assert card.matches_any_capability(["translation", "code"]) is True
    assert card.matches_any_capability(["translation", "speech"]) is False
    assert card.matches_all_capabilities(["code", "web"]) is True
    assert card.matches_all_capabilities(["code", "translation"]) is False


@pytest.mark.asyncio
async def test_a2a_discovery_cards_v3_card_registration_and_lookup():
    local_card = AgentCardV3(
        agent_id="local-agent",
        name="Local",
        description="Local agent",
        capabilities=["orchestration"],
        endpoints={"http": "http://127.0.0.1:8000"},
    )

    discovery = A2ADiscoveryCardsV3(local_card=local_card)
    assert discovery.get_agent_by_id("local-agent") == local_card
    assert len(discovery.get_all_agents()) == 1

    peer_card = AgentCardV3(
        agent_id="peer-agent",
        name="Peer",
        description="Peer agent",
        capabilities=["code_execution"],
        endpoints={"http": "http://127.0.0.1:8001"},
    )
    discovery.register_card(peer_card)

    assert discovery.get_agent_by_id("peer-agent") == peer_card
    assert len(discovery.get_all_agents()) == 2

    code_agents = discovery.find_agents_by_capability("code")
    assert len(code_agents) == 1
    assert code_agents[0].agent_id == "peer-agent"


def test_a2a_discovery_cards_v3_parse_card():
    discovery = A2ADiscoveryCardsV3()

    card_dict = {
        "agent_id": "card-1",
        "name": "Card One",
        "description": "Desc",
        "capabilities": ["llm_reasoning"],
        "endpoints": {"api": "http://127.0.0.1:9000"},
    }

    # Direct dict parse
    parsed_dict = discovery.parse_card(card_dict)
    assert parsed_dict.agent_id == "card-1"

    # Direct json string parse
    parsed_json = discovery.parse_card(json.dumps(card_dict))
    assert parsed_json.agent_id == "card-1"

    # Broadcast envelope parse (dict payload)
    envelope = {
        "type": "a2a_discovery_broadcast",
        "version": "3.0",
        "payload": card_dict,
    }
    parsed_envelope = discovery.parse_card(envelope)
    assert parsed_envelope.agent_id == "card-1"

    # Broadcast envelope parse (string payload)
    envelope_str = {
        "type": "a2a_discovery_broadcast",
        "version": "3.0",
        "payload": json.dumps(card_dict),
    }
    parsed_envelope_str = discovery.parse_card(json.dumps(envelope_str))
    assert parsed_envelope_str.agent_id == "card-1"

    # Unsupported envelope version error
    bad_version_envelope = {
        "type": "a2a_discovery_broadcast",
        "version": "9.0",
        "payload": card_dict,
    }
    with pytest.raises(ValueError, match="Unsupported envelope version"):
        discovery.parse_card(bad_version_envelope)


@pytest.mark.asyncio
async def test_a2a_discovery_cards_v3_resolve_peer_for_capability():
    local_card = AgentCardV3(
        agent_id="local",
        name="Local",
        description="Local",
        capabilities=["code_execution"],
        endpoints={},
    )
    discovery = A2ADiscoveryCardsV3(local_card=local_card)

    active_peer = AgentCardV3(
        agent_id="active-peer",
        name="Active Peer",
        description="Active",
        capabilities=["code_execution"],
        endpoints={},
        status="active",
    )
    inactive_peer = AgentCardV3(
        agent_id="inactive-peer",
        name="Inactive Peer",
        description="Inactive",
        capabilities=["code_execution"],
        endpoints={},
        status="inactive",
    )

    discovery.register_card(active_peer)
    discovery.register_card(inactive_peer)

    resolved = await discovery.resolve_peer_for_capability("code_execution")
    assert resolved is not None
    assert resolved.agent_id == "active-peer"


@pytest.mark.asyncio
async def test_fetch_and_parse_cards_mock_network():
    discovery = A2ADiscoveryCardsV3()

    card_1 = {
        "agent_id": "remote-1",
        "name": "Remote 1",
        "description": "Remote Agent One",
        "capabilities": ["image_generation"],
        "endpoints": {"mcp": "http://remote1/mcp"},
    }
    card_2 = {
        "agent_id": "remote-2",
        "name": "Remote 2",
        "description": "Remote Agent Two",
        "capabilities": ["speech_recognition"],
        "endpoints": {"mcp": "http://remote2/mcp"},
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = [card_1, card_2]

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        fetched = await discovery.fetch_and_parse_cards(
            "http://registry.a2a/cards", auth_token="valid-token"
        )

        mock_get.assert_called_once_with(
            "http://registry.a2a/cards",
            headers={"Authorization": "Bearer valid-token"},
        )
        assert len(fetched) == 2
        assert fetched[0].agent_id == "remote-1"
        assert fetched[1].agent_id == "remote-2"
        assert discovery.get_agent_by_id("remote-1") is not None


@pytest.mark.asyncio
async def test_fetch_and_parse_cards_error_handling():
    discovery = A2ADiscoveryCardsV3()

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )
        mock_get.return_value = mock_response

        with pytest.raises(httpx.HTTPStatusError):
            await discovery.fetch_and_parse_cards("http://invalid.url/cards")


@pytest.mark.asyncio
async def test_broadcast_card_mock_network():
    local_card = AgentCardV3(
        agent_id="broadcaster",
        name="Broadcaster",
        description="Broadcasting agent",
        capabilities=["discovery_broadcast"],
        endpoints={"a2a": "http://broadcaster:8000"},
    )

    discovery = A2ADiscoveryCardsV3(local_card=local_card)

    mock_response_ok = MagicMock()
    mock_response_ok.raise_for_status = MagicMock()

    mock_response_err = MagicMock()
    mock_response_err.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Internal Error", request=MagicMock(), response=MagicMock(status_code=500)
    )

    async def mock_post_impl(url, json=None):
        if url == "http://peer1/discovery":
            return mock_response_ok
        else:
            return mock_response_err

    with patch("httpx.AsyncClient.post", side_effect=mock_post_impl):
        results = await discovery.broadcast_card(
            ["http://peer1/discovery", "http://peer2/discovery"]
        )

        assert results["http://peer1/discovery"] is True
        assert results["http://peer2/discovery"] is False


def test_alias_import():
    assert A2AAgentDiscoveryV3 is A2ADiscoveryCardsV3
