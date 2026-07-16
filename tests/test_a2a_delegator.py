import pytest
import respx
import json
from httpx import Response
from typing import Generator
from magda_agent.agents.a2a_delegator import A2ADelegatorSubAgent
from magda_agent.integration.a2a_cards import AgentCardV3

@pytest.fixture
def peer_card_json() -> str:
    """
    Fixture returning a raw JSON string of a peer agent card.

    Returns:
        A JSON formatted string.
    """
    return json.dumps({
        "agent_id": "expert-data-01",
        "name": "DataExpert",
        "description": "Specialized in data analytics and statistics",
        "capabilities": ["data_analytics", "math_solving"],
        "endpoints": {"rpc": "http://expert-data-01/rpc"},
        "protocol_version": "v3"
    })

@pytest.fixture
def peer_card_json_no_rpc() -> str:
    """
    Fixture returning a raw JSON string of a peer agent card without an rpc endpoint.

    Returns:
        A JSON formatted string.
    """
    return json.dumps({
        "agent_id": "no-rpc-agent-01",
        "name": "NoRPCAgent",
        "description": "An agent without RPC endpoint",
        "capabilities": ["image_generation"],
        "endpoints": {},
        "protocol_version": "v3"
    })

def test_subagent_initialization() -> None:
    """
    Tests that the A2ADelegatorSubAgent is initialized correctly with default values.
    """
    subagent = A2ADelegatorSubAgent()
    assert subagent.agent_id == "a2a-delegator-subagent"
    assert subagent.name == "A2A Delegator Subagent"
    assert "delegation" in subagent.capabilities

    local_card = subagent.get_local_card()
    assert isinstance(local_card, AgentCardV3)
    assert local_card.agent_id == "a2a-delegator-subagent"
    assert local_card.name == "A2A Delegator Subagent"

def test_subagent_custom_initialization() -> None:
    """
    Tests that the A2ADelegatorSubAgent is initialized correctly with custom values.
    """
    subagent = A2ADelegatorSubAgent(
        agent_id="custom-delegator",
        name="Custom Coordinator",
        capabilities=["orchestration", "planning"]
    )
    assert subagent.agent_id == "custom-delegator"
    assert subagent.name == "Custom Coordinator"
    assert "orchestration" in subagent.capabilities

    local_card = subagent.get_local_card()
    assert local_card.agent_id == "custom-delegator"
    assert "planning" in local_card.capabilities

def test_discover_peers(peer_card_json: str) -> None:
    """
    Tests discovering and registering peers from raw JSON cards.
    """
    subagent = A2ADelegatorSubAgent()
    parsed_cards = subagent.discover_peers([peer_card_json])

    assert len(parsed_cards) == 1
    peer = parsed_cards[0]
    assert peer.agent_id == "expert-data-01"
    assert peer.name == "DataExpert"
    assert peer.has_capability("data_analytics")

def test_find_peers_by_capability(peer_card_json: str) -> None:
    """
    Tests finding registered peers by specific capabilities.
    """
    subagent = A2ADelegatorSubAgent()
    subagent.discover_peers([peer_card_json])

    matched = subagent.find_peers_by_capability("data_analytics")
    assert len(matched) == 1
    assert matched[0].agent_id == "expert-data-01"

    # Prefix matching test (e.g. math matches math_solving)
    prefix_matched = subagent.find_peers_by_capability("math")
    assert len(prefix_matched) == 1
    assert prefix_matched[0].agent_id == "expert-data-01"

    none_matched = subagent.find_peers_by_capability("nonexistent-skill")
    assert len(none_matched) == 0

@pytest.mark.asyncio
@respx.mock
async def test_delegate_task_success(peer_card_json: str) -> None:
    """
    Tests delegating a task to a registered peer directly.
    """
    subagent = A2ADelegatorSubAgent()
    subagent.discover_peers([peer_card_json])

    # Mock the HTTP POST to the peer's rpc endpoint
    respx.post("http://expert-data-01/rpc/delegate").mock(return_value=Response(200, json={
        "status": "success",
        "result": "Processed successfully"
    }))

    task_payload = {"action": "analyze", "data": [10, 20, 30]}
    result = await subagent.delegate_task("expert-data-01", task_payload)

    assert result == {"status": "success", "result": "Processed successfully"}

@pytest.mark.asyncio
@respx.mock
async def test_delegate_task_by_capability_success(peer_card_json: str) -> None:
    """
    Tests discovering a peer by capability and delegating a task.
    """
    subagent = A2ADelegatorSubAgent()
    subagent.discover_peers([peer_card_json])

    # Mock the HTTP POST to the peer's rpc endpoint
    respx.post("http://expert-data-01/rpc/delegate").mock(return_value=Response(200, json={
        "status": "success",
        "result": "Math problem solved"
    }))

    task_payload = {"action": "solve", "equation": "2x + 5 = 15"}
    result = await subagent.delegate_task_by_capability("math_solving", task_payload)

    assert result == {"status": "success", "result": "Math problem solved"}

@pytest.mark.asyncio
async def test_delegate_task_unknown_agent() -> None:
    """
    Tests delegating a task to an unknown agent ID.
    """
    subagent = A2ADelegatorSubAgent()
    result = await subagent.delegate_task("unknown-agent-id", {"task": "test"})
    assert result["status"] == "error"
    assert "Agent not found" in result["message"]

@pytest.mark.asyncio
async def test_delegate_task_no_rpc_endpoint(peer_card_json_no_rpc: str) -> None:
    """
    Tests delegating a task to an agent lacking an RPC endpoint.
    """
    subagent = A2ADelegatorSubAgent()
    subagent.discover_peers([peer_card_json_no_rpc])

    result = await subagent.delegate_task("no-rpc-agent-01", {"task": "test"})
    assert result["status"] == "error"
    assert "has no RPC endpoint" in result["message"]

@pytest.mark.asyncio
async def test_delegate_task_by_capability_not_found() -> None:
    """
    Tests delegating a task by capability when no peer matches.
    """
    subagent = A2ADelegatorSubAgent()
    result = await subagent.delegate_task_by_capability("nonexistent-capability", {"task": "test"})
    assert result["status"] == "error"
    assert "No peer agent found" in result["message"]

@pytest.mark.asyncio
@respx.mock
async def test_delegate_task_http_error(peer_card_json: str) -> None:
    """
    Tests handling HTTP communication errors gracefully during delegation.
    """
    subagent = A2ADelegatorSubAgent()
    subagent.discover_peers([peer_card_json])

    # Mock an HTTP 500 error
    respx.post("http://expert-data-01/rpc/delegate").mock(return_value=Response(500))

    task_payload = {"action": "analyze", "data": [10, 20, 30]}
    result = await subagent.delegate_task("expert-data-01", task_payload)

    assert result["status"] == "error"
    assert "500" in result["message"]
