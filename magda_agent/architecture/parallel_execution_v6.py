"""
Parallel Execution Manager V6.

This module provides the ParallelSubagentManagerV6 class (and alias ParallelExecutionManagerV6)
which orchestrates the concurrent execution of multiple subagents operating in isolated git worktrees,
inspired by Agent Teams isolation trends.
"""

import asyncio
import logging
from typing import List, Dict, Any, Callable, Optional, Union

from magda_agent.architecture.subagent_spawner_v5 import SubagentSpawnerV5

logger = logging.getLogger(__name__)


class ParallelSubagentManagerV6:
    """
    Manages the concurrent execution of multiple subagents operating in isolated git worktrees
    with state merging capabilities.
    """

    def __init__(self, spawner: Optional[SubagentSpawnerV5] = None) -> None:
        """
        Initialize ParallelSubagentManagerV6.

        Args:
            spawner: Optional SubagentSpawnerV5 instance. If not provided, a new one is created.
        """
        self.spawner = spawner or SubagentSpawnerV5()

    async def run_parallel_tasks(
        self,
        tasks: List[str],
        base_context: List[Dict[str, Any]],
        agent_executor_factory: Callable[[], Any],
        merge_results: bool = False
    ) -> List[Any]:
        """
        Run multiple tasks concurrently using separate subagents operating in isolated git worktrees.

        Args:
            tasks: A list of task descriptions.
            base_context: The shared conversation or execution context.
            agent_executor_factory: A callable that returns an agent executor for each task.
            merge_results: Whether to attempt git-level result merging from subagent branches.

        Returns:
            A list containing the results of each subagent's execution or Exception objects on failure.
        """
        async def create_and_spawn(task: str, idx: int) -> Any:
            executor = agent_executor_factory()
            context_copy = list(base_context)

            branch_name = f"subagent-task-{idx}" if merge_results else None

            try:
                result = await self.spawner.spawn_subagent(
                    task_description=task,
                    full_context=context_copy,
                    agent_executor=executor,
                    branch_name=branch_name,
                    merge_results=merge_results
                )
                return result
            except Exception as e:
                logger.error(f"Parallel subagent task '{task}' failed with error: {e}")
                return e

        coroutines = [create_and_spawn(task, idx) for idx, task in enumerate(tasks)]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        return list(results)

    def merge_state(self, results: List[Any]) -> Dict[str, Any]:
        """
        Merge state results from multiple subagent executions.

        Assumes dict results for state merging. Exceptions and non-dict results are ignored gracefully.

        Args:
            results: A list of results from run_parallel_tasks.

        Returns:
            A dictionary containing the merged state.
        """
        merged_state: Dict[str, Any] = {}
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Skipping result at index {idx} due to exception: {result}")
                continue

            if isinstance(result, dict):
                merged_state.update(result)
            else:
                logger.debug(f"Result at index {idx} is not a dict: {type(result)}. Skipping state merge.")

        return merged_state


# Alias for backward compatibility and alternate task specifications
ParallelExecutionManagerV6 = ParallelSubagentManagerV6
