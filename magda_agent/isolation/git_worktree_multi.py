import asyncio
import logging
from typing import List, Callable, Coroutine, Any, Optional
from magda_agent.isolation.git_worktree import GitWorktreeManager

class GitWorktreeMultiManager:
    """
    Manages concurrent multi-agent Git worktree execution.
    Ensures parallel tasks run with total context isolation in separate git worktrees,
    preventing state bleeding.
    """
    def __init__(self, base_dir: str = "/tmp/magda_worktrees_multi") -> None:
        """
        Initializes the GitWorktreeMultiManager.

        Args:
            base_dir: Base directory under which individual worktrees will be created.
        """
        self.base_dir = base_dir
        self.worktree_manager = GitWorktreeManager(base_dir=base_dir)

    async def execute_concurrently(
        self,
        tasks: List[Callable[[str], Coroutine[Any, Any, Any]]],
        branch_names: Optional[List[Optional[str]]] = None
    ) -> List[Any]:
        """
        Executes a list of asynchronous task callables concurrently, each inside an isolated git worktree.

        Args:
            tasks: A list of callables, where each callable takes the worktree path (str) as an argument
                   and returns a Coroutine.
            branch_names: Optional list of branch names corresponding to each task. If not provided or
                          less than the number of tasks, detached HEAD worktrees are created.

        Returns:
            A list of results from the tasks in the same order.
        """
        if not tasks:
            return []

        if branch_names is None:
            branch_names = [None] * len(tasks)
        elif len(branch_names) < len(tasks):
            branch_names = list(branch_names) + [None] * (len(tasks) - len(branch_names))

        async def run_single_task(
            task: Callable[[str], Coroutine[Any, Any, Any]],
            branch_name: Optional[str]
        ) -> Any:
            """
            Runs a single task within an isolated git worktree managed context.
            """
            async with self.worktree_manager.isolated_environment(branch_name=branch_name) as worktree_path:
                logging.info(f"Running concurrent task in worktree path: {worktree_path}")
                return await task(worktree_path)

        results = await asyncio.gather(*(run_single_task(t, b) for t, b in zip(tasks, branch_names)), return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logging.error(f"Concurrent task execution encountered an exception: {r}")
                raise r

        return results
