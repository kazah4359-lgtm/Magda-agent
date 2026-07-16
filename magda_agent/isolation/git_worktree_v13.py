from contextlib import asynccontextmanager
import asyncio
import logging
import os
import shutil
import uuid
from typing import Optional, AsyncGenerator, Callable, Coroutine, Any, TypeVar

T = TypeVar("T")

class GitWorktreeManagerV13:
    """
    V13 Git Worktree Manager designed to handle isolated workspaces for SubAgents.
    Ensures that multi-agent tasks do not suffer from context bleeding or file collisions
    by executing concurrent subtasks within completely isolated git worktrees.
    Provides robust lifecycle guarantees, automatic cleanup, and execution timeout protection.
    """

    def __init__(self, base_dir: str = "/tmp/magda_worktrees_v13") -> None:
        """
        Initializes the V13 Git Worktree Manager.

        Args:
            base_dir: The directory under which worktree checkouts will be created.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.active_worktrees = set()

    async def create_worktree_async(self, branch_name: Optional[str] = None) -> str:
        """
        Asynchronously creates a new git worktree. If branch_name is not provided,
        a detached worktree is created based on the current HEAD.

        Args:
            branch_name: Optional name of the branch to create or use.

        Returns:
            The absolute path to the newly created worktree.

        Raises:
            RuntimeError: If worktree creation fails.
        """
        worktree_id = str(uuid.uuid4())[:8]
        worktree_path = os.path.join(self.base_dir, f"worktree_{worktree_id}")

        if branch_name:
            cmd = ["git", "worktree", "add", "-b", branch_name, worktree_path, "HEAD"]
        else:
            cmd = ["git", "worktree", "add", "-d", worktree_path, "HEAD"]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                err_msg = stderr.decode()
                logging.error(f"Failed to create git worktree: {err_msg}")
                raise RuntimeError(f"Git worktree creation failed: {err_msg}")

            logging.info(f"Created git worktree at {worktree_path}")
            self.active_worktrees.add(worktree_path)
            return worktree_path
        except Exception as e:
            logging.error(f"Error executing git worktree add: {e}")
            raise

    async def remove_worktree_async(self, worktree_path: str) -> None:
        """
        Asynchronously removes a git worktree and deletes its directory.
        Forces the removal if there are modified or untracked files.

        Args:
            worktree_path: The directory path of the worktree to remove.
        """
        cmd = ["git", "worktree", "remove", "--force", worktree_path]
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logging.warning(f"Failed to remove git worktree via git command: {stderr.decode()}")
                # Fallback to manual directory deletion if git worktree remove fails
                if os.path.exists(worktree_path):
                    shutil.rmtree(worktree_path)
            else:
                logging.info(f"Removed git worktree at {worktree_path}")
        except Exception as e:
            logging.error(f"Error executing git worktree remove: {e}")
            if os.path.exists(worktree_path):
                shutil.rmtree(worktree_path)
        finally:
            self.active_worktrees.discard(worktree_path)

    @asynccontextmanager
    async def isolated_environment(self, branch_name: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        An async context manager to wrap code execution with worktree isolation.
        Ensures setup and automatic cleanup of the worktree.

        Args:
            branch_name: Optional branch name to create/checkout.

        Yields:
            The path to the isolated worktree directory.
        """
        worktree_path = await self.create_worktree_async(branch_name=branch_name)
        try:
            yield worktree_path
        finally:
            await self.remove_worktree_async(worktree_path)

    async def execute_in_isolation(
        self,
        task_coro_func: Callable[[str], Coroutine[Any, Any, T]],
        branch_name: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> T:
        """
        Executes a given asynchronous callable inside an isolated Git worktree.
        Guarantees that the worktree is cleaned up even if the task fails,
        raises an error, or times out.

        Args:
            task_coro_func: Async callable taking the worktree path (str) as its single argument.
            branch_name: Optional name of the branch to use for isolation.
            timeout: Optional float specifying max execution time in seconds.

        Returns:
            The result of task_coro_func.

        Raises:
            asyncio.TimeoutError: If the execution exceeds the timeout limit.
            Exception: Re-raises any exception thrown by task_coro_func.
        """
        async with self.isolated_environment(branch_name=branch_name) as worktree_path:
            task = task_coro_func(worktree_path)
            if timeout is not None:
                try:
                    return await asyncio.wait_for(task, timeout=timeout)
                except asyncio.TimeoutError:
                    logging.error(f"Task execution timed out after {timeout}s in {worktree_path}")
                    raise
            else:
                return await task
