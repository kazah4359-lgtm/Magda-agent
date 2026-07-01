"""
Parallel Subagents Manager.

This module provides the ParallelSubagentManager class which orchestrates
the concurrent execution of multiple subagents for independent parallel tasks,
inspired by the Claude Agent SDK.
"""

import asyncio
from typing import List, Dict, Any, Callable, Optional

from magda_agent.architecture.subagent_spawning import SubagentSpawner

class ParallelSubagentManager:
    """
    Manages the concurrent execution of multiple subagents.
    """

    def __init__(self, spawner: Optional[SubagentSpawner] = None) -> None:
        """
        Initialize the ParallelSubagentManager.

        Args:
            spawner: Optional SubagentSpawner instance. If not provided, a new one is created.
        """
        self.spawner = spawner or SubagentSpawner()

    async def run_parallel_tasks(
        self,
        tasks: List[str],
        base_context: List[Dict[str, Any]],
        agent_executor_factory: Callable[[], Any]
    ) -> List[Any]:
        """
        Run multiple tasks concurrently using separate subagents.

        Args:
            tasks: A list of task descriptions.
            base_context: The shared conversation or execution context.
            agent_executor_factory: A callable that returns an agent executor for each task.

        Returns:
            A list containing the results of each subagent's execution.
        """
        coroutines = []
        for task in tasks:
            executor = agent_executor_factory()

            # Use a copy of base_context to prevent race conditions during concurrent mutations
            context_copy = list(base_context)

            coro = self.spawner.spawn_subagent(
                task_description=task,
                full_context=context_copy,
                agent_executor=executor
            )
            coroutines.append(coro)

        results = await asyncio.gather(*coroutines)
        return list(results)
