import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from magda_agent.integration.a2a_discovery import AgentCard
from magda_agent.integration.a2a_protocol import A2AProtocolManager

@pytest.fixture
def mock_agent_card():
    return AgentCard(
        agent_id="test-agent-123",
        name="TestAgent",
        description="A test agent",
        capabilities=["coding"],
        endpoints={"mcp": "http://localhost:9000"}
    )

@pytest.fixture
def local_card():
    return AgentCard("local", "local", "local", [], {})

@pytest.mark.asyncio
async def test_a2a_protocol_discovery(mock_agent_card, local_card):
    manager = A2AProtocolManager(local_card)
    await manager.discover_peers([mock_agent_card.to_json()])
    peers = manager.get_known_peers()
    assert len(peers) == 1
    assert peers[0].agent_id == "test-agent-123"

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post', new_callable=AsyncMock)
async def test_a2a_protocol_delegation(mock_post, mock_agent_card, local_card):
    # Mock httpx response to behave synchronously on .json() and .raise_for_status() per guidelines
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"status": "Success"}}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    manager = A2AProtocolManager(local_card)
    await manager.discover_peers([mock_agent_card.to_json()])
    result = await manager.delegate_task("coding", {"task": "Write code"})
    # As seen in a2a_delegation.py, this matches the format "Delegated to Agent {name}: {status}"
    assert result == "Delegated to Agent TestAgent: Success"
