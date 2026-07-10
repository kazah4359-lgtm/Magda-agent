import asyncio
from typing import List, Dict, Any, Optional, Callable
import logging

class DAGMultiAgentPlanner:
    """
    A multi-agent task execution planner that handles dependency graphs (DAGs) internally
    to process sub-agent tasks in parallel when they don't have dependencies.
    """

    def __init__(self) -> None:
        """
        Initializes the DAG planner.
        """
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def add_task(self, task_id: str, payload: Any, dependencies: Optional[List[str]] = None) -> None:
        """
        Adds a task to the DAG.

        Args:
            task_id (str): Unique identifier for the task.
            payload (Any): The payload or context for the task.
            dependencies (Optional[List[str]]): List of task IDs that this task depends on.
        """
        self.tasks[task_id] = {
            "id": task_id,
            "payload": payload,
            "dependencies": dependencies or [],
            "completed": False,
            "result": None
        }

    async def execute_dag(self, mock_executor: Optional[Callable[[Any], Any]] = None) -> Dict[str, Any]:
        """
        Executes the entire DAG, running independent tasks in parallel.

        Args:
            mock_executor (Optional[Callable[[Any], Any]]): An optional async function to execute tasks.
                                                            If None, a default mock is used.

        Returns:
            Dict[str, Any]: A dictionary mapping task_id to its result.
        """
        task_futures: Dict[str, asyncio.Future] = {}
        for task_id in self.tasks:
            task_futures[task_id] = asyncio.Future()

        async def run_task(task_id: str) -> None:
            task = self.tasks[task_id]

            try:
                # Wait for all dependencies
                for dep_id in task["dependencies"]:
                    if dep_id not in task_futures:
                        raise ValueError(f"Dependency {dep_id} for task {task_id} not found in DAG.")
                    await task_futures[dep_id]

                # Execute the task
                if mock_executor:
                    result = await mock_executor(task["payload"])
                else:
                    # Default mock behavior if no executor provided
                    result = f"Result of {task_id}"

                task["completed"] = True
                task["result"] = result
                if not task_futures[task_id].done():
                    task_futures[task_id].set_result(result)
            except Exception as e:
                if not task_futures[task_id].done():
                    task_futures[task_id].set_exception(e)
                raise

        # Create all tasks
        execution_tasks = [asyncio.create_task(run_task(task_id)) for task_id in self.tasks]

        # Wait for all to complete
        if execution_tasks:
            results = await asyncio.gather(*execution_tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    # To prevent "Future exception was never retrieved", we need to ensure all futures that have an exception are read.
                    # A simple way to do this is to just call exception() on all of them here since we're going to raise anyway.
                    for future in task_futures.values():
                        if future.done() and not future.cancelled():
                            future.exception()
                    raise res

        return {task_id: self.tasks[task_id]["result"] for task_id in self.tasks}
