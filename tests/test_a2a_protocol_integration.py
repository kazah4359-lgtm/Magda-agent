import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from magda_agent.integration.a2a_discovery import AgentCard
from magda_agent.integration.a2a_protocol import A2AProtocolManager
from magda_agent.integration.a2a_protocol_integration import A2AProtocolIntegration

@pytest.fixture
def mock_agent_card():
    return AgentCard(
        agent_id="test-peer-999",
        name="TestPeer",
        description="A test peer agent",
        capabilities=["code_execution"],
        endpoints={"mcp": "http://peer.local:9000"}
    )

@pytest.fixture
def local_card():
    return AgentCard("local", "local", "local", [], {})

@pytest.mark.asyncio
async def test_a2a_protocol_integration_discovery(mock_agent_card, local_card):
    manager = A2AProtocolManager(local_card)
    integration = A2AProtocolIntegration(manager)
    peers = await integration.start_and_discover([mock_agent_card.to_json()])
    assert len(peers) == 1
    assert peers[0].agent_id == "test-peer-999"

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post', new_callable=AsyncMock)
async def test_a2a_protocol_integration_execute_task(mock_post, mock_agent_card, local_card):
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"status": "Success"}}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    manager = A2AProtocolManager(local_card)
    integration = A2AProtocolIntegration(manager)
    await integration.start_and_discover([mock_agent_card.to_json()])
    result = await integration.execute_task_with_peer("code_execution", {"task": "test"})
    assert result == "Delegated to Agent TestPeer: Success"
