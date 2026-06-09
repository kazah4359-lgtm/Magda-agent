import pytest
from unittest.mock import patch, MagicMock
from magda_agent.integration.a2a_security import A2ASecurityContext
from magda_agent.integration.a2a_discovery import A2ADiscovery, AgentCard
from magda_agent.integration.a2a_delegation import A2ADelegator

def test_security_context():
    """Test token generation, validation and tracing."""
    ctx = A2ASecurityContext()
    token = ctx.generate_token()
    assert token.startswith("a2a_")
    assert ctx.validate_token(token)
    assert not ctx.validate_token("invalid_token")

    trace_id = ctx.trace_action("test_action", {"key": "value"})
    assert isinstance(trace_id, str)
    assert len(trace_id) > 0

@pytest.mark.asyncio
async def test_discovery_fetch_with_auth():
    """Test that fetch_cards requires and validates auth tokens."""
    card = AgentCard(agent_id="1", name="A1", description="D1", capabilities=["code"], endpoints={"mcp": "http://test"})
    ctx = A2ASecurityContext()
    discovery = A2ADiscovery(local_card=card, security_context=ctx)

    token = ctx.generate_token()
    await discovery.fetch_cards([], auth_token=token)

    with pytest.raises(ValueError, match="Invalid authentication token"):
        await discovery.fetch_cards([], auth_token="invalid_token")

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_delegator_with_auth(mock_post):
    """Test that delegator adds token to request headers."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"result": {"status": "Success"}}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    card = AgentCard(agent_id="1", name="A1", description="D1", capabilities=["code"], endpoints={"mcp": "http://test"})
    ctx = A2ASecurityContext()
    discovery = A2ADiscovery(local_card=card, security_context=ctx)
    discovery._discovered_agents["1"] = card
    discovery._capability_index["code"] = ["1"]

    delegator = A2ADelegator(discovery)

    res = await delegator.delegate_subplan("code", {"task": "do_stuff"})
    assert "Success" in res

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert "headers" in kwargs
    assert "Authorization" in kwargs["headers"]
    assert kwargs["headers"]["Authorization"].startswith("Bearer a2a_")
