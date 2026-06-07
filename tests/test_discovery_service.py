import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from magda_agent.integration.discovery import A2ADiscoveryService

@pytest.fixture
def discovery_service():
    return A2ADiscoveryService(endpoint="http://mock-discovery:8000")

@pytest.mark.asyncio
@patch("httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_discover_peers_success(mock_get, discovery_service):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = [
        {"id": "agent_alpha", "name": "Alpha Agent", "capabilities": ["math", "logic"]},
        {"id": "agent_beta", "name": "Beta Agent", "capabilities": ["search"]}
    ]
    mock_get.return_value = mock_response

    peers = await discovery_service.discover_peers()

    assert len(peers) == 2
    assert discovery_service.get_peer_card("agent_alpha")["name"] == "Alpha Agent"
    assert discovery_service.get_peer_card("agent_beta")["capabilities"] == ["search"]
    mock_get.assert_called_once_with("http://mock-discovery:8000/cards")

@pytest.mark.asyncio
@patch("httpx.AsyncClient.get", new_callable=AsyncMock)
async def test_discover_peers_failure(mock_get, discovery_service):
    mock_get.side_effect = Exception("Network Error")

    peers = await discovery_service.discover_peers()

    assert peers == []
    assert len(discovery_service.peers) == 0

def test_get_peer_card_not_found(discovery_service):
    assert discovery_service.get_peer_card("nonexistent_agent") is None
