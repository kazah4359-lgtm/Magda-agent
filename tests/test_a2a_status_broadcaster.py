import pytest
import respx
import httpx
import json
from magda_agent.integration.a2a_discovery import AgentCard
from magda_agent.integration.a2a_status_broadcaster import A2AStatusBroadcaster

@pytest.fixture
def agent_card():
    return AgentCard(
        agent_id="test-agent-123",
        name="TestAgent",
        description="A test agent",
        capabilities=["test-cap"],
        endpoints={"rpc": "http://localhost:8000"}
    )

@pytest.fixture
def broadcaster(agent_card):
    return A2AStatusBroadcaster(agent_card, "http://test-registry/broadcast")

def test_generate_status_payload(broadcaster, agent_card):
    """Test that the status payload correctly serializes the AgentCard and status info."""
    payload = broadcaster.generate_status_payload(is_available=True, active_tasks=2)

    assert "agent_card" in payload
    assert "status" in payload

    assert payload["agent_card"]["agent_id"] == "test-agent-123"
    assert payload["agent_card"]["name"] == "TestAgent"

    assert payload["status"]["is_available"] is True
    assert payload["status"]["active_tasks"] == 2

@pytest.mark.asyncio
@respx.mock
async def test_broadcast_status_success(broadcaster):
    """Test successful status broadcast."""
    request = respx.post("http://test-registry/broadcast").mock(return_value=httpx.Response(200))

    result = await broadcaster.broadcast_status(is_available=True, active_tasks=1)

    assert result is True
    assert request.called

    # Verify the payload sent
    sent_payload = json.loads(request.calls.last.request.content)
    assert sent_payload["status"]["is_available"] is True
    assert sent_payload["status"]["active_tasks"] == 1
    assert sent_payload["agent_card"]["agent_id"] == "test-agent-123"

@pytest.mark.asyncio
@respx.mock
async def test_broadcast_status_http_error(broadcaster):
    """Test broadcast status handles HTTP errors gracefully."""
    request = respx.post("http://test-registry/broadcast").mock(return_value=httpx.Response(500))

    result = await broadcaster.broadcast_status(is_available=True, active_tasks=1)

    assert result is False
    assert request.called

@pytest.mark.asyncio
@respx.mock
async def test_broadcast_status_network_error(broadcaster):
    """Test broadcast status handles network errors resiliently."""
    request = respx.post("http://test-registry/broadcast").mock(side_effect=httpx.NetworkError("Network down"))

    result = await broadcaster.broadcast_status(is_available=True, active_tasks=1)

    assert result is False
    assert request.called
