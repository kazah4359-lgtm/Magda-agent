import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.integration.a2a_peer_routing_v2 import A2APeerRouterV2, TaskEvaluationResult
from magda_agent.integration.a2a_cards import AgentCardV4, A2ADiscoveryV4
from magda_agent.integration.a2a_delegator_v4 import A2ADelegatorV4


def test_evaluate_complexity_simple() -> None:
    """
    Test that a simple task gets a low complexity score and is not marked complex.
    """
    router = A2APeerRouterV2(complexity_threshold=5.0)
    task_payload = {
        "id": "task-101",
        "description": "Short description",
        "risk": "low"
    }

    eval_result = router.evaluate_complexity(task_payload)
    assert eval_result.task_id == "task-101"
    assert eval_result.is_complex is False
    assert eval_result.complexity_score < 5.0


def test_evaluate_complexity_complex() -> None:
    """
    Test that a task with subtasks, long description, and high risk gets a high complexity score.
    """
    router = A2APeerRouterV2(complexity_threshold=5.0)
    task_payload = {
        "id": "task-102",
        "description": "A " * 300,  # > 500 chars
        "subtasks": ["sub1", "sub2", "sub3"],
        "risk": "high",
        "required_capabilities": ["code_execution"]
    }

    eval_result = router.evaluate_complexity(task_payload)
    assert eval_result.task_id == "task-102"
    assert eval_result.is_complex is True
    assert eval_result.complexity_score >= 5.0
    assert eval_result.required_capabilities == ["code_execution"]


def test_find_matching_peer() -> None:
    """
    Test capability matching using A2ADiscoveryV4 and AgentCardV4.
    """
    local_card = AgentCardV4(
        agent_id="local-1",
        name="Local Agent",
        description="Main agent",
        capabilities=["general"],
        endpoints={"rpc": "http://localhost:8000/rpc"}
    )
    discovery = A2ADiscoveryV4(local_card=local_card)

    peer_card = AgentCardV4(
        agent_id="peer-1",
        name="Peer Coder",
        description="Specialized coding agent",
        capabilities=["code_execution", "refactoring"],
        endpoints={"rpc": "http://peer-agent:8000/rpc"}
    )
    discovery._register_agent(peer_card)

    router = A2APeerRouterV2(discovery=discovery)

    matched = router.find_matching_peer(["code_execution"])
    assert matched is not None
    assert matched.agent_id == "peer-1"

    no_match = router.find_matching_peer(["quantum_computing"])
    assert no_match is None


@pytest.mark.asyncio
async def test_route_and_delegate_local_execution() -> None:
    """
    Test that simple tasks with no matching peer requirement execute locally.
    """
    router = A2APeerRouterV2(complexity_threshold=5.0)
    task_payload = {
        "id": "task-simple",
        "description": "Simple task",
        "risk": "low"
    }

    res = await router.route_and_delegate(task_payload)
    assert res["status"] == "local_execution"
    assert res["task_id"] == "task-simple"


@pytest.mark.asyncio
async def test_route_and_delegate_successful() -> None:
    """
    Test that complex tasks or tasks requiring capabilities are routed and delegated to matched peer.
    """
    local_card = AgentCardV4(
        agent_id="local-1",
        name="Local Agent",
        description="Main agent",
        capabilities=["general"],
        endpoints={"rpc": "http://localhost:8000/rpc"}
    )
    discovery = A2ADiscoveryV4(local_card=local_card)

    peer_card = AgentCardV4(
        agent_id="peer-specialist",
        name="Data Specialist",
        description="Data analytics peer",
        capabilities=["data_analysis"],
        endpoints={"rpc": "http://data-peer:8000/rpc"}
    )
    discovery._register_agent(peer_card)

    delegator = A2ADelegatorV4()

    with patch.object(delegator, "delegate_task", new_callable=AsyncMock) as mock_delegate:
        mock_delegate.return_value = {"status": "completed", "output": "analytics_done"}

        router = A2APeerRouterV2(discovery=discovery, delegator=delegator, complexity_threshold=5.0)

        task_payload = {
            "id": "task-analytics",
            "description": "Perform complex dataset analysis",
            "required_capabilities": ["data_analysis"],
            "is_complex": True
        }

        res = await router.route_and_delegate(task_payload)

        assert res["status"] == "delegated"
        assert res["peer_id"] == "peer-specialist"
        assert res["result"] == {"status": "completed", "output": "analytics_done"}

        mock_delegate.assert_called_once()
        _, kwargs = mock_delegate.call_args
        assert kwargs["peer_endpoint"] == "http://data-peer:8000/rpc"
        assert kwargs["payload"]["task_id"] == "task-analytics"


@pytest.mark.asyncio
async def test_route_and_delegate_no_matching_peer_raises() -> None:
    """
    Test that routing raises ValueError when required capabilities cannot be matched by any peer.
    """
    local_card = AgentCardV4(
        agent_id="local-1",
        name="Local Agent",
        description="Main agent",
        capabilities=["general"],
        endpoints={"rpc": "http://localhost:8000/rpc"}
    )
    discovery = A2ADiscoveryV4(local_card=local_card)
    router = A2APeerRouterV2(discovery=discovery, complexity_threshold=5.0)

    task_payload = {
        "id": "task-unsupported",
        "description": "Complex task requiring missing capability",
        "required_capabilities": ["nonexistent_capability"],
        "risk": "high"
    }

    with pytest.raises(ValueError, match="No peer agent found matching required capabilities"):
        await router.route_and_delegate(task_payload)
