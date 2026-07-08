import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from magda_agent.integration.a2a_workflow import A2AWorkflowManager
from magda_agent.integration.a2a_cards import AgentCardV3
from magda_agent.integration.a2a_security import A2ASecurityContext
from magda_agent.integration.a2a_delegation import A2ADelegator

@pytest.fixture
def mock_delegator():
    delegator = MagicMock(spec=A2ADelegator)
    delegator.delegate_to_peer = AsyncMock()
    return delegator

@pytest.fixture
def workflow_manager(mock_delegator):
    return A2AWorkflowManager(delegator=mock_delegator)

@pytest.fixture
def target_agents():
    agent1 = AgentCardV3(
        agent_id="agent-1",
        name="Step 1 Agent",
        description="First agent",
        capabilities=["data_gathering"],
        endpoints={"mcp": "http://agent1:8000/mcp"},

    )
    agent2 = AgentCardV3(
        agent_id="agent-2",
        name="Step 2 Agent",
        description="Second agent",
        capabilities=["data_processing"],
        endpoints={"mcp": "http://agent2:8000/mcp"},

    )
    return [agent1, agent2]

@pytest.mark.asyncio
async def test_execute_chain_successful(workflow_manager, mock_delegator, target_agents):
    mock_delegator.delegate_to_peer.side_effect = ["Success Step 1", "Success Step 2"]

    workflow_steps = [
        {"name": "step_1", "action": "gather_data"},
        {"name": "step_2", "action": "process_data"}
    ]
    initial_context = {"input": "start_data"}

    final_context = await workflow_manager.execute_chain(workflow_steps, target_agents, initial_context)

    # Verify delegation calls
    assert mock_delegator.delegate_to_peer.call_count == 2

    # Check the context was updated correctly
    assert final_context["input"] == "start_data"
    assert final_context["result_step_1"] == "Success Step 1"
    assert final_context["result_step_2"] == "Success Step 2"

@pytest.mark.asyncio
async def test_execute_chain_length_mismatch(workflow_manager, target_agents):
    workflow_steps = [
        {"name": "step_1", "action": "gather_data"}
    ]
    initial_context = {"input": "start_data"}

    with pytest.raises(ValueError, match="The number of workflow steps must match the number of target agents."):
        await workflow_manager.execute_chain(workflow_steps, target_agents, initial_context)

@pytest.mark.asyncio
async def test_execute_chain_failure(workflow_manager, mock_delegator, target_agents):
    mock_delegator.delegate_to_peer.side_effect = [
        "Success Step 1",
        Exception("Network error on step 2")
    ]

    workflow_steps = [
        {"name": "step_1", "action": "gather_data"},
        {"name": "step_2", "action": "process_data"}
    ]
    initial_context = {"input": "start_data"}

    final_context = await workflow_manager.execute_chain(workflow_steps, target_agents, initial_context)

    # It should have called both, but step 2 failed
    assert mock_delegator.delegate_to_peer.call_count == 2

    # Check context
    assert final_context["input"] == "start_data"
    assert final_context["result_step_1"] == "Success Step 1"
    assert "result_step_2" not in final_context
    assert final_context["error"] == "Network error on step 2"
    assert final_context["failed_step"] == "step_2"
