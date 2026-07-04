import pytest
import json
from magda_agent.integration.a2a import A2AManager
from magda_agent.integration.a2a_cards import AgentCardV3

@pytest.fixture
def local_card_v3():
    return AgentCardV3(
        agent_id="agent-v3-001",
        name="MagdaLocalV3",
        description="Local agent V3 for testing",
        capabilities=["chat", "code_execution"],
        endpoints={"rpc": "http://localhost:8080/rpc"}
    )

@pytest.fixture
def remote_card_v3():
    return AgentCardV3(
        agent_id="agent-remote-v3-001",
        name="RemoteWorkerV3",
        description="Worker node V3",
        capabilities=["image_generation", "chat_advanced"],
        endpoints={"rpc": "http://192.168.1.10:8080/rpc"}
    )

@pytest.mark.asyncio
async def test_a2a_manager_init_v3(local_card_v3):
    manager = A2AManager(local_card=local_card_v3)
    assert manager.discovery.__class__.__name__ == "A2ADiscoveryV3"

@pytest.mark.asyncio
async def test_a2a_manager_discover_peers_v3(local_card_v3, remote_card_v3):
    manager = A2AManager(local_card=local_card_v3)

    envelope = {
        "type": "a2a_discovery_broadcast",
        "version": "3.0",
        "payload": remote_card_v3.to_json()
    }

    await manager.discover_peers([json.dumps(envelope)])

    known_peers = manager.get_known_peers()
    assert len(known_peers) == 1
    assert known_peers[0].agent_id == "agent-remote-v3-001"
