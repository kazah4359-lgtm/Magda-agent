import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from magda_agent.integration.a2a_discovery import A2ADiscovery, AgentCard
from magda_agent.integration.a2a_delegation import A2ADelegator
from magda_agent.agents.planner_agent import PlannerAgent

@pytest.fixture
def mock_agent_card():
    return AgentCard(
        agent_id="test-agent-123",
        name="TestAgent",
        description="A test agent",
        capabilities=["coding", "analysis"],
        endpoints={"mcp": "http://localhost:9000"}
    )

@pytest.fixture
def a2a_discovery(mock_agent_card):
    # local card doesn't matter for finding remote agents here
    local_card = AgentCard("local", "local", "local", [], {})
    discovery = A2ADiscovery(local_card)
    # manually inject the mock agent
    discovery._discovered_agents[mock_agent_card.agent_id] = mock_agent_card
    discovery._capability_index["coding"] = [mock_agent_card.agent_id]
    discovery._capability_index["analysis"] = [mock_agent_card.agent_id]
    return discovery

@pytest.fixture
def a2a_delegator(a2a_discovery):
    return A2ADelegator(a2a_discovery)

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post', new_callable=AsyncMock)
async def test_a2a_delegator_finds_agent(mock_post, a2a_delegator):
    from unittest.mock import MagicMock
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"status": "Success"}}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = await a2a_delegator.delegate_subplan("coding", {"task": "Write hello world"})
    assert result == "Delegated to Agent TestAgent: Success"

@pytest.mark.asyncio
async def test_a2a_delegator_no_agent(a2a_delegator):
    result = await a2a_delegator.delegate_subplan("drawing", {"task": "Draw a cat"})
    assert result == "No agent found"

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post', new_callable=AsyncMock)
async def test_planner_agent_delegation(mock_post, a2a_delegator):
    # Test PlannerAgent integrates with A2ADelegator
    planner_agent = PlannerAgent(planner=None, a2a_delegator=a2a_delegator)

    from unittest.mock import MagicMock
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"status": "Success"}}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = await planner_agent.delegate_subplan("coding", {"task": "Refactor module"})
    assert result == "Delegated to Agent TestAgent: Success"

@pytest.mark.asyncio
async def test_planner_agent_no_delegator():
    # Test PlannerAgent gracefully handles missing delegator
    planner_agent = PlannerAgent(planner=None, a2a_delegator=None)

    result = await planner_agent.delegate_subplan("coding", {"task": "Refactor module"})
    assert result == "Delegation failed: No A2ADelegator configured."

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post', new_callable=AsyncMock)
async def test_planner_agent_plan_delegation(mock_post, a2a_delegator):
    from unittest.mock import MagicMock
    mock_planner = MagicMock()
    mock_planner.get_current_plan.return_value = [
        {"id": "step_1", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "coding"}, "description": "Delegate coding task"}
    ]
    planner_agent = PlannerAgent(planner=mock_planner, a2a_delegator=a2a_delegator)

    from unittest.mock import MagicMock
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"status": "Success"}}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    plan = await planner_agent.plan("do coding task")
    assert plan[0]["result"] == "Delegated to Agent TestAgent: Success"


@pytest.mark.asyncio
async def test_a2a_delegator_split_plan(a2a_delegator):
    plan = [
        {"id": "1", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "coding"}, "description": "code it"},
        {"id": "2", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "analysis"}, "description": "analyze it"},
        {"id": "3", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "coding"}, "description": "code again"},
        {"id": "4", "skill": "some_other_skill", "description": "do something else"}
    ]
    split = a2a_delegator.split_plan(plan)
    assert len(split) == 3
    assert split[0]["capability"] == "coding"
    assert split[1]["capability"] == "analysis"
    assert split[2]["capability"] == "coding"

@pytest.mark.asyncio
async def test_a2a_delegator_split_plan_missing_capability(a2a_delegator):
    plan = [
        {"id": "1", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "coding"}, "description": "code it"},
        {"id": "2", "skill": "delegate_to_agent", "skill_kwargs": {}, "description": "missing capability"},
        {"id": "3", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "analysis"}, "description": "analyze it"}
    ]
    split = a2a_delegator.split_plan(plan)
    assert len(split) == 2
    assert split[0]["capability"] == "coding"
    assert split[0]["steps"][0]["id"] == "1"
    assert split[1]["capability"] == "analysis"
    assert split[1]["steps"][0]["id"] == "3"

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post', new_callable=AsyncMock)
async def test_a2a_delegator_execute_plan(mock_post, a2a_delegator):
    plan = [
        {"id": "1", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "coding"}, "description": "code it"},
        {"id": "2", "skill": "some_other_skill", "description": "do something else"}
    ]

    from unittest.mock import MagicMock
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"status": "Success"}}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    results = await a2a_delegator.execute_plan(plan)
    assert "1" in results
    assert "2" not in results
    assert results["1"] == "Delegated to Agent TestAgent: Success"


@pytest.mark.asyncio
@patch('httpx.AsyncClient.post', new_callable=AsyncMock)
async def test_a2a_delegator_delegate_to_peer(mock_post, a2a_delegator, mock_agent_card):
    from unittest.mock import MagicMock
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"status": "Success"}}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    # Need to clear tracer registry to test recording
    from magda_agent.integration.a2a_tracing import A2ATracer
    A2ATracer.clear_registry()
    trace_id = "test_peer_delegation_trace"
    A2ATracer.set_trace_id(trace_id)

    result = await a2a_delegator.delegate_to_peer(mock_agent_card, {"task": "Direct delegation"})
    assert result == "Delegated to Peer Agent TestAgent: Success"

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert "X-A2A-Trace-ID" in kwargs["headers"]

    history = A2ATracer.get_trace(trace_id)
    # Events should be "delegation_sent" from inject_headers and "peer_delegation"
    events = [e["event"] for e in history]
    assert "peer_delegation" in events

    # Check details of peer_delegation
    peer_event = next(e for e in history if e["event"] == "peer_delegation")
    assert peer_event["details"]["target_agent_id"] == mock_agent_card.agent_id

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post', new_callable=AsyncMock)
async def test_a2a_delegator_delegate_to_peer_no_endpoint(mock_post, a2a_delegator):
    from magda_agent.integration.a2a_discovery import AgentCard
    no_endpoint_card = AgentCard("no-endpoint-id", "NoEndpointAgent", "Desc", ["coding"], {})

    result = await a2a_delegator.delegate_to_peer(no_endpoint_card, {"task": "Direct delegation"})
    assert result == "Agent NoEndpointAgent missing MCP endpoint"
    mock_post.assert_not_called()

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post', new_callable=AsyncMock)
async def test_a2a_delegator_delegate_to_peer_failure(mock_post, a2a_delegator, mock_agent_card):
    mock_post.side_effect = Exception("Network error")

    result = await a2a_delegator.delegate_to_peer(mock_agent_card, {"task": "Direct delegation"})
    assert result == "Delegation to peer TestAgent failed: Network error"
