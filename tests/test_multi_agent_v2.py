import pytest
import asyncio
from magda_agent.agents.multi_agent_v2 import DAGMultiAgentPlanner

@pytest.mark.asyncio
async def test_dag_multi_agent_planner_execution():
    """
    Test that the DAGMultiAgentPlanner executes tasks in the correct order based on dependencies.
    """
    planner = DAGMultiAgentPlanner()

    planner.add_task("task1", "payload1")
    planner.add_task("task2", "payload2", dependencies=["task1"])
    planner.add_task("task3", "payload3", dependencies=["task1"])
    planner.add_task("task4", "payload4", dependencies=["task2", "task3"])

    execution_order = []

    async def mock_executor(payload):
        # Record execution order based on payload
        execution_order.append(payload)
        await asyncio.sleep(0.01) # Simulate some work to allow event loop to switch context
        return f"processed_{payload}"

    results = await planner.execute_dag(mock_executor=mock_executor)

    # task1 must be first
    assert execution_order[0] == "payload1"

    # task2 and task3 can be in any order, but must be in positions 1 and 2
    assert set(execution_order[1:3]) == {"payload2", "payload3"}

    # task4 must be last
    assert execution_order[3] == "payload4"

    assert results["task1"] == "processed_payload1"
    assert results["task2"] == "processed_payload2"
    assert results["task3"] == "processed_payload3"
    assert results["task4"] == "processed_payload4"

@pytest.mark.asyncio
async def test_dag_multi_agent_planner_missing_dependency():
    """
    Test that the DAGMultiAgentPlanner raises an error when a dependency is missing.
    """
    planner = DAGMultiAgentPlanner()
    planner.add_task("task1", "payload1", dependencies=["missing_task"])

    with pytest.raises(ValueError, match="Dependency missing_task for task task1 not found in DAG."):
        try:
            await planner.execute_dag()
        finally:
            # Clean up pending tasks to avoid "Task was destroyed but it is pending"
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()

@pytest.mark.asyncio
async def test_dag_multi_agent_planner_default_executor():
    """
    Test that the DAGMultiAgentPlanner uses a default executor when none is provided.
    """
    planner = DAGMultiAgentPlanner()
    planner.add_task("task1", "payload1")

    results = await planner.execute_dag()
    assert results["task1"] == "Result of task1"


@pytest.mark.asyncio
async def test_dag_multi_agent_planner_exception_handling():
    """
    Test that the DAGMultiAgentPlanner correctly handles exceptions in tasks and prevents deadlocks.
    """
    planner = DAGMultiAgentPlanner()

    planner.add_task("task1", "payload1")
    planner.add_task("task2", "payload2", dependencies=["task1"])

    async def mock_executor_with_error(payload):
        if payload == "payload1":
            raise RuntimeError("Task failed")
        return f"processed_{payload}"

    with pytest.raises(RuntimeError, match="Task failed"):
        try:
            await planner.execute_dag(mock_executor=mock_executor_with_error)
        finally:
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()
