import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from magda_agent.integration.a2a_discovery import A2ADiscovery, AgentCard
from magda_agent.integration.a2a_delegation import A2ADelegator
from magda_agent.integration.a2a_orchestrator import A2AOrchestrator

@pytest.fixture
def a2a_discovery():
    local_card = AgentCard("local", "local", "local", [], {})
    discovery = A2ADiscovery(local_card)
    return discovery

@pytest.fixture
def a2a_delegator(a2a_discovery):
    delegator = A2ADelegator(a2a_discovery)
    return delegator

@pytest.fixture
def a2a_orchestrator(a2a_discovery, a2a_delegator):
    return A2AOrchestrator(a2a_discovery, a2a_delegator)


@pytest.mark.asyncio
async def test_dispatch_concurrently(a2a_orchestrator):
    # Mock the delegator's delegate_subplan method
    a2a_orchestrator.delegator.delegate_subplan = AsyncMock()

    async def mock_delegate(capability, step):
        # Simulate some delay to test concurrency properly
        await asyncio.sleep(0.01)
        return f"Delegated {step['id']} for {capability}"

    a2a_orchestrator.delegator.delegate_subplan.side_effect = mock_delegate

    sub_plans = [
        {"capability": "coding", "steps": [{"id": "step_1"}]},
        {"capability": "analysis", "steps": [{"id": "step_2"}, {"id": "step_3"}]}
    ]

    results = await a2a_orchestrator.dispatch_concurrently(sub_plans)

    assert len(results) == 3
    assert results["step_1"] == "Delegated step_1 for coding"
    assert results["step_2"] == "Delegated step_2 for analysis"
    assert results["step_3"] == "Delegated step_3 for analysis"
    assert a2a_orchestrator.delegator.delegate_subplan.call_count == 3


@pytest.mark.asyncio
async def test_dispatch_concurrently_with_exception(a2a_orchestrator, caplog):
    # Mock the delegator's delegate_subplan method
    a2a_orchestrator.delegator.delegate_subplan = AsyncMock()

    async def mock_delegate(capability, step):
        if step['id'] == "step_error":
            raise Exception("Simulated delegation failure")
        return f"Delegated {step['id']} for {capability}"

    a2a_orchestrator.delegator.delegate_subplan.side_effect = mock_delegate

    sub_plans = [
        {"capability": "coding", "steps": [{"id": "step_1"}, {"id": "step_error"}]},
    ]

    results = await a2a_orchestrator.dispatch_concurrently(sub_plans)

    # step_1 should succeed, step_error should fail and return exception
    assert len(results) == 1
    assert results["step_1"] == "Delegated step_1 for coding"
    assert "Simulated delegation failure" in caplog.text


@pytest.mark.asyncio
async def test_execute_orchestrated_plan_sequential(a2a_orchestrator):
    plan = [
        {"id": "step_1", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "coding"}, "description": "code it"}
    ]

    a2a_orchestrator.delegator.execute_plan = AsyncMock(return_value={"step_1": "Sequential Success"})

    results = await a2a_orchestrator.execute_orchestrated_plan(plan, concurrent=False)

    a2a_orchestrator.delegator.execute_plan.assert_called_once_with(plan)
    assert results["step_1"] == "Sequential Success"

@pytest.mark.asyncio
async def test_execute_orchestrated_plan_concurrent(a2a_orchestrator):
    plan = [
        {"id": "step_1", "skill": "delegate_to_agent", "skill_kwargs": {"capability": "coding"}, "description": "code it"}
    ]

    # Mock the concurrent dispatch
    a2a_orchestrator.dispatch_concurrently = AsyncMock(return_value={"step_1": "Concurrent Success"})
    # Need to make sure split_plan is used
    a2a_orchestrator.delegator.split_plan = MagicMock(return_value=[{"capability": "coding", "steps": [plan[0]]}])

    results = await a2a_orchestrator.execute_orchestrated_plan(plan, concurrent=True)

    a2a_orchestrator.dispatch_concurrently.assert_called_once()
    assert results["step_1"] == "Concurrent Success"

@pytest.mark.asyncio
async def test_execute_orchestrated_plan_empty(a2a_orchestrator):
    results = await a2a_orchestrator.execute_orchestrated_plan([])
    assert results == {}
