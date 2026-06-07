import pytest
from unittest.mock import AsyncMock, patch
from magda_agent.integration.a2a_discovery import A2ADiscovery, AgentCard
from magda_agent.integration.a2a_delegation import A2ADelegator
import httpx

@pytest.fixture
def mock_agent_card():
    return AgentCard(
        agent_id="test-agent-123",
        name="TestAgent",
        description="A test agent",
        capabilities=["coding", "analysis"],
        endpoints={"mcp": "http://localhost:9000"}
    )

@pytest.fixture
def a2a_discovery(mock_agent_card):
    # local card doesn't matter for finding remote agents here
    local_card = AgentCard("local", "local", "local", [], {})
    discovery = A2ADiscovery(local_card)
    # manually inject the mock agent
    discovery._discovered_agents[mock_agent_card.agent_id] = mock_agent_card
    discovery._capability_index["coding"] = [mock_agent_card.agent_id]
    discovery._capability_index["analysis"] = [mock_agent_card.agent_id]
    return discovery

@pytest.fixture
def a2a_delegator(a2a_discovery):
    return A2ADelegator(a2a_discovery)

@pytest.mark.asyncio
async def test_a2a_delegator_finds_agent_success(a2a_delegator):
    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        # For an AsyncMock, when the code does `await client.post()`, it gets what is set as return_value.
        # But our code then calls `response.raise_for_status()` and `response.json()`
        # which means response itself can just be a regular Mock.
        from unittest.mock import MagicMock
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"status": "Success"}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = await a2a_delegator.delegate_subplan("coding", {"task": "Write hello world"})
        assert result == "Delegated to Agent TestAgent: Success"
        mock_post.assert_called_once()

@pytest.mark.asyncio
async def test_a2a_delegator_finds_agent_failure(a2a_delegator):
    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.RequestError("Network error")

        result = await a2a_delegator.delegate_subplan("coding", {"task": "Write hello world"})
        assert result == "Delegation to TestAgent failed: Network error"
        mock_post.assert_called_once()

@pytest.mark.asyncio
async def test_a2a_delegator_no_agent(a2a_delegator):
    result = await a2a_delegator.delegate_subplan("drawing", {"task": "Draw a cat"})
    assert result == "No agent found"

@pytest.mark.asyncio
async def test_a2a_delegator_missing_endpoint(a2a_discovery):
    agent_without_endpoint = AgentCard(
        agent_id="test-agent-456",
        name="BadAgent",
        description="A bad test agent",
        capabilities=["cooking"],
        endpoints={}
    )
    a2a_discovery._discovered_agents[agent_without_endpoint.agent_id] = agent_without_endpoint
    a2a_discovery._capability_index["cooking"] = [agent_without_endpoint.agent_id]

    delegator = A2ADelegator(a2a_discovery)
    result = await delegator.delegate_subplan("cooking", {"task": "Bake a cake"})
    assert result == "Agent BadAgent missing MCP endpoint"
