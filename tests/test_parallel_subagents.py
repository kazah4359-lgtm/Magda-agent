import asyncio
import pytest
from unittest.mock import AsyncMock
from magda_agent.architecture.parallel_subagents import ParallelSubagentManager
from magda_agent.architecture.subagent_spawning import SubagentSpawner

class MockExecutor:
    """Mock agent executor that simulates async work."""
    def __init__(self, delay: float, result_prefix: str):
        self.delay = delay
        self.result_prefix = result_prefix

    async def execute(self, context: list) -> str:
        await asyncio.sleep(self.delay)
        # SubagentSpawner appends exactly {"role": "user", "content": f"Task: {task_description}"}
        # to the context.
        task_desc_str = context[-1]["content"] if context else "unknown task"
        return f"{self.result_prefix}: {task_desc_str}"


@pytest.mark.asyncio
async def test_parallel_execution_time():
    """
    Test that run_parallel_tasks actually runs concurrently.
    If sequential, time would be delay * N. If parallel, it should be ~delay.
    """
    manager = ParallelSubagentManager()
    tasks = ["Task 1", "Task 2", "Task 3"]
    base_context = [{"role": "system", "content": "base"}]

    # 0.1s delay for each task
    delay = 0.1

    def factory():
        return MockExecutor(delay=delay, result_prefix="Done")

    start_time = asyncio.get_event_loop().time()
    results = await manager.run_parallel_tasks(tasks, base_context, factory)
    end_time = asyncio.get_event_loop().time()

    elapsed = end_time - start_time

    assert len(results) == 3
    assert "Done: Task: Task 1" in results
    assert "Done: Task: Task 2" in results
    assert "Done: Task: Task 3" in results

    # Check if execution time is much less than sequential time (0.3s)
    # Give some margin for overhead, but it shouldn't take 0.3s.
    assert elapsed < 0.2, f"Execution took too long ({elapsed}s), might be sequential."


@pytest.mark.asyncio
async def test_empty_tasks():
    manager = ParallelSubagentManager()
    results = await manager.run_parallel_tasks([], [], lambda: None)
    assert results == []

@pytest.mark.asyncio
async def test_callable_executor():
    manager = ParallelSubagentManager()

    async def mock_callable(ctx):
        return "callable done"

    # The SubagentSpawner supports both an object with an .execute method and a raw callable.
    results = await manager.run_parallel_tasks(["Task X"], [], lambda: mock_callable)
    assert len(results) == 1
    assert results[0] == "callable done"

@pytest.mark.asyncio
async def test_base_context_race_condition():
    """
    Test that modifying the context inside SubagentSpawner does not affect the original base_context.
    """
    manager = ParallelSubagentManager()
    tasks = ["Task 1", "Task 2"]
    base_context = [{"role": "system", "content": "base"}]

    def factory():
        return MockExecutor(delay=0, result_prefix="Done")

    await manager.run_parallel_tasks(tasks, base_context, factory)

    # base_context should remain unchanged
    assert len(base_context) == 1
