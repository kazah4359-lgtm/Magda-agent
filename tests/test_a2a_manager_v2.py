import pytest
import json
from unittest.mock import AsyncMock
from magda_agent.integration.a2a import A2AManager
from magda_agent.integration.a2a_discovery_v2 import AgentCardV2

@pytest.fixture
def local_card_v2() -> AgentCardV2:
    """Fixture for a local AgentCardV2."""
    return AgentCardV2(
        agent_id="agent-001",
        name="MagdaLocal",
        description="Local agent for testing",
        capabilities=["chat", "planning"],
        endpoints={"rpc": "http://localhost:8080/rpc"}
    )

@pytest.fixture
def remote_card_v2() -> AgentCardV2:
    """Fixture for a remote AgentCardV2."""
    return AgentCardV2(
        agent_id="agent-remote-001",
        name="RemoteWorker",
        description="Worker node",
        capabilities=["code_execution", "linting"],
        endpoints={"rpc": "http://192.168.1.10:8080/rpc"}
    )

@pytest.mark.asyncio
async def test_a2a_manager_start_v2(local_card_v2: AgentCardV2) -> None:
    """Test that A2AManager correctly broadcasts its local capabilities using AgentCardV2."""
    manager = A2AManager(local_card=local_card_v2)
    broadcast_json = await manager.start()

    envelope = json.loads(broadcast_json)
    assert envelope["type"] == "a2a_discovery_broadcast"

    payload = json.loads(envelope["payload"])
    assert payload["agent_id"] == "agent-001"
    assert "planning" in payload["capabilities"]

@pytest.mark.asyncio
async def test_a2a_manager_discover_and_delegate_v2(local_card_v2: AgentCardV2, remote_card_v2: AgentCardV2) -> None:
    """Test that A2AManager discovers peers and delegates subplans correctly using AgentCardV2."""
    manager = A2AManager(local_card=local_card_v2)

    network_envelopes = [
        json.dumps({"type": "a2a_discovery_broadcast", "version": "2.0", "payload": remote_card_v2.to_json()})
    ]
    await manager.discover_peers(mock_network_cards=network_envelopes)

    peers = manager.get_known_peers()
    assert len(peers) == 1
    assert peers[0].agent_id == "agent-remote-001"

    # Test delegation to found peer
    manager.delegator.delegate_subplan = AsyncMock(return_value="Delegated to Agent RemoteWorker") # type: ignore
    result = await manager.delegate_task("code_execution", {"code": "print('hello')"})
    assert result == f"Delegated to Agent RemoteWorker"

    # Test delegation to missing capability
    manager.delegator.delegate_subplan = AsyncMock(return_value="No agent found") # type: ignore
    result_missing = await manager.delegate_task("image_generation", {"prompt": "cat"})
    assert result_missing == "No agent found"
