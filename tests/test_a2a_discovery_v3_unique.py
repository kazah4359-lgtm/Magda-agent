import pytest
import json
import respx
import httpx
from unittest.mock import AsyncMock

from magda_agent.integration.a2a_cards import AgentCardV3
from magda_agent.integration.a2a_discovery_v3_unique import A2ADiscoveryServiceV3Unique

@pytest.fixture
def valid_card() -> AgentCardV3:
    """Fixture providing a valid AgentCardV3."""
    return AgentCardV3(
        agent_id="agent-001",
        name="TestAgent",
        description="A test agent",
        capabilities=["code_execution", "chat"],
        endpoints={"rpc": "http://localhost:8080/rpc"},
        protocol_version="v3"
    )

def test_parse_and_register_valid_cards(valid_card: AgentCardV3) -> None:
    """Tests parsing and registering a valid AgentCard."""
    service = A2ADiscoveryServiceV3Unique()
    valid_json = valid_card.to_json()

    parsed_cards = service.parse_and_register_cards([valid_json])

    assert len(parsed_cards) == 1
    assert parsed_cards[0].agent_id == "agent-001"

    registered = service.get_agent_card("agent-001")
    assert registered is not None
    assert registered.name == "TestAgent"

def test_parse_invalid_cards() -> None:
    """Tests parsing invalid JSON cards."""
    service = A2ADiscoveryServiceV3Unique()

    invalid_json = '{"agent_id": "missing_quotes}'
    missing_fields_json = '{"agent_id": "test-agent", "name": "Agent"}'

    parsed_cards = service.parse_and_register_cards([invalid_json, missing_fields_json])

    assert len(parsed_cards) == 0

def test_find_agents_by_capability(valid_card: AgentCardV3) -> None:
    """Tests finding an agent by an existing capability."""
    service = A2ADiscoveryServiceV3Unique()
    service.parse_and_register_cards([valid_card.to_json()])

    agents = service.find_agents_by_capability("code_execution")
    assert len(agents) == 1
    assert agents[0].agent_id == "agent-001"

    agents = service.find_agents_by_capability("unknown_cap")
    assert len(agents) == 0

@pytest.mark.asyncio
@respx.mock
async def test_delegate_task_success(valid_card: AgentCardV3) -> None:
    """Tests successfully delegating a task to a peer agent."""
    service = A2ADiscoveryServiceV3Unique()
    service.parse_and_register_cards([valid_card.to_json()])

    task_payload = {"task": "do_something", "data": "test"}
    rpc_endpoint = valid_card.endpoints["rpc"]

    route = respx.post(f"{rpc_endpoint}/delegate").mock(return_value=httpx.Response(200, json={"status": "accepted"}))

    response = await service.delegate_task("agent-001", task_payload)

    assert response == {"status": "accepted"}
    assert route.called

@pytest.mark.asyncio
async def test_delegate_task_agent_not_found() -> None:
    """Tests delegating to a non-existent agent raises ValueError."""
    service = A2ADiscoveryServiceV3Unique()

    with pytest.raises(ValueError, match="Agent not found"):
        await service.delegate_task("unknown-agent", {})

@pytest.mark.asyncio
async def test_delegate_task_no_endpoint(valid_card: AgentCardV3) -> None:
    """Tests delegating to an agent without an RPC endpoint raises ValueError."""
    service = A2ADiscoveryServiceV3Unique()
    valid_card.endpoints = {}
    service.parse_and_register_cards([valid_card.to_json()])

    with pytest.raises(ValueError, match="has no RPC endpoint"):
        await service.delegate_task("agent-001", {})

@pytest.mark.asyncio
@respx.mock
async def test_delegate_task_http_error(valid_card: AgentCardV3) -> None:
    """Tests delegating raises httpx.HTTPStatusError on failure."""
    service = A2ADiscoveryServiceV3Unique()
    service.parse_and_register_cards([valid_card.to_json()])

    rpc_endpoint = valid_card.endpoints["rpc"]
    respx.post(f"{rpc_endpoint}/delegate").mock(return_value=httpx.Response(500))

    with pytest.raises(httpx.HTTPStatusError):
        await service.delegate_task("agent-001", {})
