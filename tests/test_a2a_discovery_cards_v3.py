import pytest
import respx
import httpx
from typing import List, Dict, Union, Any
from magda_agent.integration.a2a_discovery_cards_v3 import A2ADiscoveryCardsV3
from magda_agent.integration.a2a_cards import AgentCardV3
import json

@pytest.fixture
def valid_cards_data() -> List[Union[Dict[str, Any], str]]:
    """
    Fixture providing valid and invalid test data for agent cards.
    """
    return [
        {
            "agent_id": "test-agent-1",
            "name": "Test Agent 1",
            "description": "A test agent",
            "capabilities": ["chat", "code"],
            "endpoints": {"rpc": "http://test-agent-1/rpc"},
            "protocol_version": "v3"
        },
        json.dumps({
            "agent_id": "test-agent-2",
            "name": "Test Agent 2",
            "description": "Another test agent",
            "capabilities": ["search"],
            "endpoints": {"rpc": "http://test-agent-2/rpc"},
            "protocol_version": "v3"
        }),
        "invalid json string",
        {"invalid": "dictionary_without_required_fields"}
    ]

@pytest.mark.asyncio
@respx.mock
async def test_fetch_and_parse_cards(valid_cards_data: List[Union[Dict[str, Any], str]]) -> None:
    """
    Test that fetch_and_parse_cards correctly handles HTTP endpoints
    and parses mixed valid and invalid JSON representations.
    """
    endpoint_url = "http://discovery-service/api/cards"

    # Mock the endpoint
    respx.get(endpoint_url).mock(return_value=httpx.Response(200, json=valid_cards_data))

    discovery = A2ADiscoveryCardsV3()

    cards = await discovery.fetch_and_parse_cards(endpoint_url)

    assert len(cards) == 2

    assert cards[0].agent_id == "test-agent-1"
    assert cards[0].name == "Test Agent 1"
    assert cards[0].capabilities == ["chat", "code"]
    assert cards[0].protocol_version == "v3"

    assert cards[1].agent_id == "test-agent-2"
    assert cards[1].name == "Test Agent 2"
    assert cards[1].capabilities == ["search"]
    assert cards[1].protocol_version == "v3"

@pytest.mark.asyncio
@respx.mock
async def test_fetch_and_parse_cards_http_error() -> None:
    """
    Test that fetch_and_parse_cards raises an HTTPStatusError when the endpoint fails.
    """
    endpoint_url = "http://discovery-service/api/cards"

    respx.get(endpoint_url).mock(return_value=httpx.Response(500))

    discovery = A2ADiscoveryCardsV3()

    with pytest.raises(httpx.HTTPStatusError):
        await discovery.fetch_and_parse_cards(endpoint_url)

@pytest.mark.asyncio
@respx.mock
async def test_fetch_and_parse_cards_invalid_response_format() -> None:
    """
    Test that fetch_and_parse_cards gracefully handles responses that are not lists.
    """
    endpoint_url = "http://discovery-service/api/cards"

    # Return a dict instead of a list
    respx.get(endpoint_url).mock(return_value=httpx.Response(200, json={"error": "not a list"}))

    discovery = A2ADiscoveryCardsV3()

    cards = await discovery.fetch_and_parse_cards(endpoint_url)

    assert cards == []
