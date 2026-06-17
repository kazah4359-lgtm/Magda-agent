import pytest
import asyncio
from unittest.mock import patch, MagicMock
from magda_agent.integration.a2a_tracing import A2ATracer, TRACE_HEADER
from magda_agent.integration.a2a_delegation import A2ADelegator
from magda_agent.integration.a2a_discovery import AgentCard, A2ADiscovery
from magda_agent.integration.a2a_server import A2AServer
from magda_agent.planning.planner import Planner

def test_a2a_tracer_unit():
    """Test A2ATracer basic functionality."""
    A2ATracer.set_trace_id(None)
    assert A2ATracer.get_current_trace_id() is None

    trace_id = A2ATracer.generate_trace_id()
    A2ATracer.set_trace_id(trace_id)
    assert A2ATracer.get_current_trace_id() == trace_id

    headers = {}
    A2ATracer.inject_headers(headers)
    assert headers[TRACE_HEADER] == trace_id

    # Case-insensitive extraction
    assert A2ATracer.extract_from_headers({"x-a2a-trace-id": "123"}) == "123"
    assert A2ATracer.extract_from_headers({"X-A2A-TRACE-ID": "456"}) == "456"

@pytest.mark.asyncio
async def test_a2a_tracer_context_isolation():
    """Test that trace IDs are isolated per context/task."""
    async def task_with_trace(tid):
        A2ATracer.set_trace_id(tid)
        await asyncio.sleep(0.1)
        return A2ATracer.get_current_trace_id()

    t1 = task_with_trace("trace_1")
    t2 = task_with_trace("trace_2")

    results = await asyncio.gather(t1, t2)
    assert results == ["trace_1", "trace_2"]

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_delegator_injects_trace(mock_post):
    """Test that A2ADelegator injects the trace header."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"result": {"status": "accepted"}}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    card = AgentCard(agent_id="1", name="A1", description="D1", capabilities=["code"], endpoints={"mcp": "http://test"})
    discovery = A2ADiscovery(local_card=card)
    discovery._discovered_agents["1"] = card
    discovery._capability_index["code"] = ["1"]

    delegator = A2ADelegator(discovery)

    # Pre-set trace ID
    A2ATracer.set_trace_id("test_distributed_trace")

    await delegator.delegate_subplan("code", {"foo": "bar"})

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["headers"][TRACE_HEADER] == "test_distributed_trace"

@pytest.mark.asyncio
async def test_server_extracts_trace():
    """Test that A2AServer extracts the trace header from incoming requests."""
    from unittest.mock import AsyncMock
    planner = MagicMock(spec=Planner)
    planner.generate_plan = AsyncMock()
    server = A2AServer(planner=planner)

    # Mock FastAPI Request
    request = MagicMock()
    request.headers = {TRACE_HEADER: "incoming_trace_id"}
    request.json = AsyncMock(return_value={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "delegate_task",
        "params": {"task": "do something"}
    })

    A2ATracer.set_trace_id(None)
    await server.handle_request(request)

    assert A2ATracer.get_current_trace_id() == "incoming_trace_id"
    planner.generate_plan.assert_called_once()
