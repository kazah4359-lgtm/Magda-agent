import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.planning.delegation import AgentDelegationProtocol

@pytest.mark.asyncio
async def test_execute_delegated_steps_success():
    mock_planner = MagicMock()
    mock_planner.spawn_sub_agent = AsyncMock(return_value="Sub-agent success")

    protocol = AgentDelegationProtocol(planner=mock_planner)

    steps = [
        {"id": "step1", "description": "Do task 1"},
        {"id": "step2", "description": "Do task 2"}
    ]
    context = "test context"

    results = await protocol.execute_delegated_steps(steps, context)

    assert len(results) == 2
    assert results["step1"] == "Sub-agent success"
    assert results["step2"] == "Sub-agent success"

    assert mock_planner.spawn_sub_agent.call_count == 2
    mock_planner.spawn_sub_agent.assert_any_call(task="Do task 1", context="test context")
    mock_planner.spawn_sub_agent.assert_any_call(task="Do task 2", context="test context")


@pytest.mark.asyncio
async def test_execute_delegated_steps_with_error():
    mock_planner = MagicMock()
    # First call succeeds, second call fails
    mock_planner.spawn_sub_agent = AsyncMock(side_effect=["Success", Exception("Failed task")])

    protocol = AgentDelegationProtocol(planner=mock_planner)

    steps = [
        {"id": "step1", "description": "Do task 1"},
        {"id": "step2", "description": "Do task 2"}
    ]

    results = await protocol.execute_delegated_steps(steps, "ctx")

    assert len(results) == 2
    assert results["step1"] == "Success"
    assert results["step2"] == "Error: Failed task"

@pytest.mark.asyncio
async def test_execute_delegated_steps_invalid_steps():
    mock_planner = MagicMock()
    mock_planner.spawn_sub_agent = AsyncMock(return_value="Success")

    protocol = AgentDelegationProtocol(planner=mock_planner)

    steps = [
        {"description": "Missing ID"},
        {"id": "step2"}, # Missing description
        {"id": "step3", "description": "Valid step"}
    ]

    results = await protocol.execute_delegated_steps(steps, "ctx")

    assert len(results) == 1
    assert "step3" in results
    assert results["step3"] == "Success"
