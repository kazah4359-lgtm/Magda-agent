import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from typing import AsyncGenerator, Dict, Any, List

from magda_agent.integration.a2a_cards import AgentCardV3
from magda_agent.integration.a2a_orchestration import A2AOrchestratorStream
from magda_agent.integration.a2a_security import A2ASecurityContext

@pytest.fixture
def target_agents() -> List[AgentCardV3]:
    """Provides a list of dummy target agent cards."""
    return [
        AgentCardV3(
            agent_id="agent-1",
            name="Worker1",
            description="Worker agent 1",
            capabilities=["math"],
            endpoints={"rpc": "http://10.0.0.1:8080/rpc"}
        ),
        AgentCardV3(
            agent_id="agent-2",
            name="Worker2",
            description="Worker agent 2",
            capabilities=["search"],
            endpoints={"rpc": "http://10.0.0.2:8080/rpc"}
        )
    ]

@pytest.mark.asyncio
async def test_broadcast_plan_stream_success(target_agents: List[AgentCardV3]) -> None:
    """Tests successful broadcast of a plan stream to multiple agents."""
    orchestrator = A2AOrchestratorStream(timeout=10.0)

    # Mock the delegator to return specific streams based on the agent name
    async def mock_stream_delegation_v2(agent: AgentCardV3, plan_context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        if agent.name == "Worker1":
            yield {"status": "processing", "progress": 50}
            await asyncio.sleep(0.01)
            yield {"status": "completed", "result": "Worker1_done"}
        elif agent.name == "Worker2":
            await asyncio.sleep(0.01)
            yield {"status": "started"}
            yield {"status": "completed", "result": "Worker2_done"}

    with patch.object(orchestrator.delegator, 'stream_delegation_v2', side_effect=mock_stream_delegation_v2):
        chunks = []
        async for item in orchestrator.broadcast_plan_stream(target_agents, {"task": "do_work"}):
            chunks.append(item)

        assert len(chunks) == 4
        # Since it's concurrent, we just verify all expected chunks are present
        worker1_chunks = [c["chunk"] for c in chunks if c["agent_name"] == "Worker1"]
        worker2_chunks = [c["chunk"] for c in chunks if c["agent_name"] == "Worker2"]

        assert len(worker1_chunks) == 2
        assert worker1_chunks[0]["status"] == "processing"
        assert worker1_chunks[1]["result"] == "Worker1_done"

        assert len(worker2_chunks) == 2
        assert worker2_chunks[0]["status"] == "started"
        assert worker2_chunks[1]["result"] == "Worker2_done"

@pytest.mark.asyncio
async def test_broadcast_plan_stream_with_error(target_agents: List[AgentCardV3]) -> None:
    """Tests broadcast when one agent stream raises an exception."""
    orchestrator = A2AOrchestratorStream(timeout=10.0)

    async def mock_stream_delegation_v2(agent: AgentCardV3, plan_context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        if agent.name == "Worker1":
            yield {"status": "processing"}
            raise Exception("Network failure")
        elif agent.name == "Worker2":
            yield {"status": "completed"}

    with patch.object(orchestrator.delegator, 'stream_delegation_v2', side_effect=mock_stream_delegation_v2):
        chunks = []
        async for item in orchestrator.broadcast_plan_stream(target_agents, {"task": "do_work"}):
            chunks.append(item)

        worker1_chunks = [c["chunk"] for c in chunks if c["agent_name"] == "Worker1"]
        worker2_chunks = [c["chunk"] for c in chunks if c["agent_name"] == "Worker2"]

        assert len(worker1_chunks) == 2
        assert worker1_chunks[0]["status"] == "processing"
        assert "error" in worker1_chunks[1]
        assert "Network failure" in worker1_chunks[1]["error"]

        assert len(worker2_chunks) == 1
        assert worker2_chunks[0]["status"] == "completed"

@pytest.mark.asyncio
async def test_broadcast_plan_stream_empty_agents() -> None:
    """Tests broadcast with an empty list of agents."""
    orchestrator = A2AOrchestratorStream()
    chunks = []
    async for item in orchestrator.broadcast_plan_stream([], {"task": "do_work"}):
        chunks.append(item)
    assert len(chunks) == 0
