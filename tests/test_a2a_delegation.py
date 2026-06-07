import pytest
from unittest.mock import MagicMock, AsyncMock
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
async def test_a2a_delegator_finds_agent(a2a_delegator):
    result = await a2a_delegator.delegate_subplan("coding", {"task": "Write hello world"})
    assert result == "Delegated to Agent TestAgent"

@pytest.mark.asyncio
async def test_a2a_delegator_no_agent(a2a_delegator):
    result = await a2a_delegator.delegate_subplan("drawing", {"task": "Draw a cat"})
    assert result == "No agent found"

@pytest.mark.asyncio
async def test_planner_agent_delegation(a2a_delegator):
    # Test PlannerAgent integrates with A2ADelegator
    planner_agent = PlannerAgent(planner=None, a2a_delegator=a2a_delegator)

    result = await planner_agent.delegate_subplan("coding", {"task": "Refactor module"})
    assert result == "Delegated to Agent TestAgent"

@pytest.mark.asyncio
async def test_planner_agent_no_delegator():
    # Test PlannerAgent gracefully handles missing delegator
    planner_agent = PlannerAgent(planner=None, a2a_delegator=None)

    result = await planner_agent.delegate_subplan("coding", {"task": "Refactor module"})
    assert result == "Delegation failed: No A2ADelegator configured."

@pytest.mark.asyncio
async def test_planner_agent_plan_delegation(a2a_delegator):
    mock_planner = MagicMock()
    mock_planner.get_current_plan.return_value = [
        {"id": "step_1", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "coding"}, "description": "Delegate coding task"}
    ]
    planner_agent = PlannerAgent(planner=mock_planner, a2a_delegator=a2a_delegator)

    plan = await planner_agent.plan("do coding task")
    assert plan[0]["result"] == "Delegated to Agent TestAgent"
