import pytest
import respx
import httpx
from unittest.mock import MagicMock

from magda_agent.integration.a2a_delegation_v5 import A2ADelegatorV5
from magda_agent.integration.a2a_discovery_v3_unique import A2ADiscoveryServiceV3Unique
from magda_agent.integration.a2a_cards import AgentCardV3


@pytest.fixture
def mock_agent_card() -> AgentCardV3:
    return AgentCardV3(
        agent_id="test-agent-123",
        name="Test Agent",
        description="A test agent",
        capabilities=["code_execution"],
        endpoints={"rpc": "http://test-agent/rpc"},
        protocol_version="v3"
    )


@pytest.fixture
def discovery_service(mock_agent_card: AgentCardV3) -> A2ADiscoveryServiceV3Unique:
    service = A2ADiscoveryServiceV3Unique()
    service._discovered_agents[mock_agent_card.agent_id] = mock_agent_card
    return service


@pytest.mark.asyncio
@respx.mock
async def test_delegate_task_success(discovery_service: A2ADiscoveryServiceV3Unique) -> None:
    delegator = A2ADelegatorV5(discovery_service=discovery_service)

    task_payload = {"task": "do_something"}
    expected_response = {"status": "success", "result": "done"}

    # Mock the HTTP POST request triggered by A2ADiscoveryServiceV3Unique.delegate_task
    respx.post("http://test-agent/rpc/delegate").mock(
        return_value=httpx.Response(200, json=expected_response)
    )

    result = await delegator.delegate_task("code_execution", task_payload)
    assert result == expected_response


@pytest.mark.asyncio
async def test_delegate_task_no_agents_found(discovery_service: A2ADiscoveryServiceV3Unique) -> None:
    delegator = A2ADelegatorV5(discovery_service=discovery_service)

    task_payload = {"task": "do_something"}

    # Capability not present in discovery service
    with pytest.raises(ValueError, match="No agents found with capability: nonexistent_capability"):
        await delegator.delegate_task("nonexistent_capability", task_payload)
