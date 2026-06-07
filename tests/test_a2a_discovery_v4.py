import pytest
import json
from unittest.mock import AsyncMock
from magda_agent.integration.a2a import A2AManager
from magda_agent.integration.a2a_discovery import AgentCard

@pytest.fixture
def local_card():
    return AgentCard(
        agent_id="agent-001",
        name="MagdaLocal",
        description="Local agent for testing",
        capabilities=["chat", "planning"],
        endpoints={"rpc": "http://localhost:8080/rpc"}
    )

@pytest.fixture
def remote_card():
    return AgentCard(
        agent_id="agent-remote-001",
        name="RemoteWorker",
        description="Worker node",
        capabilities=["code_execution", "linting"],
        endpoints={"rpc": "http://192.168.1.10:8080/rpc"}
    )

@pytest.mark.asyncio
async def test_a2a_manager_start(local_card):
    manager = A2AManager(local_card=local_card)
    broadcast_json = await manager.start()

    data = json.loads(broadcast_json)
    assert data["agent_id"] == "agent-001"
    assert "planning" in data["capabilities"]

@pytest.mark.asyncio
async def test_a2a_manager_discover_and_delegate(local_card, remote_card):
    manager = A2AManager(local_card=local_card)

    mock_network_cards = [remote_card.to_json()]
    await manager.discover_peers(mock_network_cards=mock_network_cards)

    peers = manager.get_known_peers()
    assert len(peers) == 1
    assert peers[0].agent_id == "agent-remote-001"

    # Test delegation to found peer
    manager.delegator.delegate_subplan = AsyncMock(return_value="Delegated to Agent RemoteWorker")
    result = await manager.delegate_task("code_execution", {"code": "print('hello')"})
    assert result == f"Delegated to Agent RemoteWorker"

    # Test delegation to missing capability
    manager.delegator.delegate_subplan = AsyncMock(return_value="No agent found")
    result_missing = await manager.delegate_task("image_generation", {"prompt": "cat"})
    assert result_missing == "No agent found"
