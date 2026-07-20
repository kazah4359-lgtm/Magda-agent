import pytest
import httpx
import respx
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Any, Dict

from magda_agent.integration.a2a_delegation_v5 import (
    AgentCardV5,
    A2ADiscoveryRegistryV5,
    A2ADelegatorV5,
)
from magda_agent.integration.a2a_security import A2ASecurityContext


@pytest.fixture
def mock_agent_card_v5() -> AgentCardV5:
    return AgentCardV5(
        agent_id="test-agent-v5",
        name="Test Agent V5",
        description="A test agent supporting v5",
        capabilities=["code_execution", "image_processing"],
        endpoints={"rpc": "http://test-agent-v5/rpc", "mcp": "http://test-agent-v5/mcp"},
        protocol_version="v5"
    )


@pytest.fixture
def mock_security_context() -> A2ASecurityContext:
    context = MagicMock(spec=A2ASecurityContext)
    context.generate_token.return_value = "fake-token-v5"
    return context


def test_agent_card_v5_serialization(mock_agent_card_v5: AgentCardV5) -> None:
    """Test JSON serialization and deserialization of AgentCardV5."""
    json_str = mock_agent_card_v5.to_json()
    deserialized = AgentCardV5.from_json(json_str)

    assert deserialized.agent_id == mock_agent_card_v5.agent_id
    assert deserialized.name == mock_agent_card_v5.name
    assert deserialized.description == mock_agent_card_v5.description
    assert deserialized.capabilities == mock_agent_card_v5.capabilities
    assert deserialized.endpoints == mock_agent_card_v5.endpoints
    assert deserialized.protocol_version == "v5"


def test_agent_card_v5_capabilities(mock_agent_card_v5: AgentCardV5) -> None:
    """Test capability matching logic including prefixes."""
    assert mock_agent_card_v5.has_capability("code_execution")
    assert mock_agent_card_v5.has_capability("code")  # prefix match
    assert not mock_agent_card_v5.has_capability("text")

    assert mock_agent_card_v5.matches_any_capability(["translation", "code_execution"])
    assert not mock_agent_card_v5.matches_any_capability(["translation"])


def test_discovery_registry_v5(mock_agent_card_v5: AgentCardV5) -> None:
    """Test registering, unregistering, parsing, and retrieving Agent Cards."""
    registry = A2ADiscoveryRegistryV5()
    registry.register_agent(mock_agent_card_v5)

    assert registry.get_agent_card("test-agent-v5") == mock_agent_card_v5
    assert mock_agent_card_v5 in registry.get_all_agents()

    # Capability search
    matched = registry.find_agents_by_capability("code")
    assert len(matched) == 1
    assert matched[0] == mock_agent_card_v5

    # Unregister
    registry.unregister_agent("test-agent-v5")
    assert registry.get_agent_card("test-agent-v5") is None


def test_discovery_registry_parsing(mock_agent_card_v5: AgentCardV5) -> None:
    """Test parsing and bulk registration from JSON strings."""
    registry = A2ADiscoveryRegistryV5()
    json_str = mock_agent_card_v5.to_json()

    parsed = registry.parse_and_register_cards([json_str, "invalid-json"])
    assert len(parsed) == 1
    assert parsed[0].agent_id == mock_agent_card_v5.agent_id
    assert registry.get_agent_card(mock_agent_card_v5.agent_id) is not None


@pytest.mark.asyncio
async def test_delegate_task_success(mock_agent_card_v5: AgentCardV5, mock_security_context: A2ASecurityContext) -> None:
    """Test successful task delegation using mocked AsyncClient post."""
    delegator = A2ADelegatorV5(security_context=mock_security_context)
    task_payload = {"task": "execute_code"}
    expected_response = {"status": "success", "result": "done"}

    mock_response = MagicMock()
    mock_response.json.return_value = expected_response
    mock_response.raise_for_status = MagicMock()

    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await delegator.delegate_task(mock_agent_card_v5, task_payload)

        assert result == expected_response
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://test-agent-v5/rpc"
        assert kwargs['json'] == task_payload
        assert 'Authorization' in kwargs['headers']
        assert kwargs['headers']['Authorization'] == "Bearer fake-token-v5"
        assert kwargs['timeout'] == 10.0


@pytest.mark.asyncio
async def test_delegate_by_capability(mock_agent_card_v5: AgentCardV5) -> None:
    """Test auto-discovery and delegation based on capability."""
    registry = A2ADiscoveryRegistryV5()
    registry.register_agent(mock_agent_card_v5)

    delegator = A2ADelegatorV5(discovery_registry=registry)
    task_payload = {"task": "image_processing"}
    expected_response = {"status": "processed"}

    with respx.mock:
        respx.post("http://test-agent-v5/rpc").mock(
            return_value=httpx.Response(200, json=expected_response)
        )

        result = await delegator.delegate_by_capability("image", task_payload)
        assert result == expected_response


@pytest.mark.asyncio
async def test_delegate_by_capability_not_found() -> None:
    """Test delegation fails when no agent has the requested capability."""
    registry = A2ADiscoveryRegistryV5()
    delegator = A2ADelegatorV5(discovery_registry=registry)

    with pytest.raises(ValueError, match="No peer agents found supporting capability"):
        await delegator.delegate_by_capability("non_existent_capability", {"task": "do_it"})


@pytest.mark.asyncio
async def test_delegate_task_no_endpoint() -> None:
    """Test delegation fails when endpoints are missing."""
    card = AgentCardV5(
        agent_id="test-endpointless",
        name="No Endpoints",
        description="Endpointless agent",
        capabilities=["nothing"],
        endpoints={},
        protocol_version="v5"
    )
    delegator = A2ADelegatorV5()

    with pytest.raises(ValueError, match="missing endpoint"):
        await delegator.delegate_task(card, {"task": "test"})


@pytest.mark.asyncio
async def test_delegate_task_http_error(mock_agent_card_v5: AgentCardV5) -> None:
    """Test delegation propagates HTTP status errors correctly."""
    delegator = A2ADelegatorV5()
    task_payload = {"task": "fail_me"}

    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.HTTPStatusError("Bad Gateway", request=MagicMock(), response=MagicMock())

        with pytest.raises(httpx.HTTPError):
            await delegator.delegate_task(mock_agent_card_v5, task_payload)
