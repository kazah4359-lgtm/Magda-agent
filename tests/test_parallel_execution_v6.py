"""
Tests for Parallel Execution Manager V6 (ParallelSubagentManagerV6).
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from magda_agent.architecture.parallel_execution_v6 import (
    ParallelSubagentManagerV6,
    ParallelExecutionManagerV6
)
from magda_agent.architecture.subagent_spawner_v5 import SubagentSpawnerV5


def test_alias_equivalence():
    """Verify ParallelExecutionManagerV6 is an alias of ParallelSubagentManagerV6."""
    assert ParallelExecutionManagerV6 is ParallelSubagentManagerV6


@pytest.mark.asyncio
async def test_run_parallel_tasks_success():
    """Test running parallel tasks successfully with mocked SubagentSpawnerV5."""
    mock_spawner = MagicMock(spec=SubagentSpawnerV5)

    async def mock_spawn_subagent(task_description, full_context, agent_executor, **kwargs):
        if "task1" in task_description:
            return {"status": "ok", "task": 1}
        elif "task2" in task_description:
            return {"status": "ok", "task": 2}
        return {"status": "unknown"}

    mock_spawner.spawn_subagent = AsyncMock(side_effect=mock_spawn_subagent)

    manager = ParallelSubagentManagerV6(spawner=mock_spawner)
    tasks = ["Execute task1", "Execute task2"]
    base_context = [{"role": "system", "content": "System context"}]

    def executor_factory():
        return MagicMock()

    results = await manager.run_parallel_tasks(
        tasks=tasks,
        base_context=base_context,
        agent_executor_factory=executor_factory,
        merge_results=True
    )

    assert len(results) == 2
    assert results[0] == {"status": "ok", "task": 1}
    assert results[1] == {"status": "ok", "task": 2}
    assert mock_spawner.spawn_subagent.call_count == 2


@pytest.mark.asyncio
async def test_run_parallel_tasks_with_failures():
    """Test handling of task failures in run_parallel_tasks."""
    mock_spawner = MagicMock(spec=SubagentSpawnerV5)

    async def mock_spawn_subagent(task_description, full_context, agent_executor, **kwargs):
        if "fail" in task_description:
            raise ValueError("Subagent execution error")
        return {"result": "success"}

    mock_spawner.spawn_subagent = AsyncMock(side_effect=mock_spawn_subagent)

    manager = ParallelSubagentManagerV6(spawner=mock_spawner)
    tasks = ["Run safe task", "Run fail task"]
    base_context = []

    def executor_factory():
        return MagicMock()

    results = await manager.run_parallel_tasks(
        tasks=tasks,
        base_context=base_context,
        agent_executor_factory=executor_factory,
        merge_results=False
    )

    assert len(results) == 2
    assert results[0] == {"result": "success"}
    assert isinstance(results[1], ValueError)
    assert str(results[1]) == "Subagent execution error"


def test_merge_state():
    """Test state merging logic for dict results and handling non-dict/exceptions."""
    manager = ParallelSubagentManagerV6()

    results = [
        {"key1": "val1", "shared": 1},
        ValueError("Failed task"),
        {"key2": "val2", "shared": 2},
        "Non-dict result",
        123
    ]

    merged = manager.merge_state(results)
    assert merged == {"key1": "val1", "key2": "val2", "shared": 2}


def test_merge_state_empty():
    """Test state merging with empty results."""
    manager = ParallelSubagentManagerV6()
    merged = manager.merge_state([])
    assert merged == {}
