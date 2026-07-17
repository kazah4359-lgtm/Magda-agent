import pytest
import json
from magda_agent.integration.a2a import A2AManager
from magda_agent.integration.a2a_discovery_v4 import AgentCardV4

@pytest.fixture
def local_card_v4() -> AgentCardV4:
    """Fixture for local AgentCardV4."""
    return AgentCardV4(
        agent_id="agent-001",
        name="MagdaLocalV4",
        description="Local agent for testing",
        capabilities=["chat", "planning"],
        endpoints={"rpc": "http://localhost:8080/rpc"}
    )

@pytest.fixture
def remote_card_v4() -> AgentCardV4:
    """Fixture for remote AgentCardV4."""
    return AgentCardV4(
        agent_id="agent-remote-001",
        name="RemoteWorkerV4",
        description="Worker node",
        capabilities=["code_execution", "linting"],
        endpoints={"rpc": "http://192.168.1.10:8080/rpc"}
    )

@pytest.mark.asyncio
async def test_a2a_manager_start_v4(local_card_v4: AgentCardV4) -> None:
    """Test A2AManager start functionality with V4 card."""
    manager = A2AManager(local_card=local_card_v4)
    broadcast_json = await manager.start()

    data = json.loads(broadcast_json)
    assert data["agent_id"] == "agent-001"
    assert "planning" in data["capabilities"]

@pytest.mark.asyncio
async def test_a2a_manager_discover_and_delegate_v4(local_card_v4: AgentCardV4, remote_card_v4: AgentCardV4) -> None:
    """Test A2AManager discovery and delegation with V4 cards."""
    manager = A2AManager(local_card=local_card_v4)

    mock_network_cards = [remote_card_v4.to_json()]
    await manager.discover_peers(mock_network_cards=mock_network_cards)

    peers = manager.get_known_peers()
    assert len(peers) == 2  # Includes local card registered upon init

    remote_peers = [p for p in peers if p.agent_id == "agent-remote-001"]
    assert len(remote_peers) == 1

    # Test delegation to found peer
    result = await manager.delegate_task("code_execution", {"code": "print('hello')"})
    assert result == "Delegated to Agent RemoteWorkerV4"

    # Test delegation to missing capability
    result_missing = await manager.delegate_task("image_generation", {"prompt": "cat"})
    assert result_missing == "No agent found"
