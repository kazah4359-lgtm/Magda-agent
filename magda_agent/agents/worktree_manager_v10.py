from contextlib import asynccontextmanager
import asyncio
import logging
import os
import shutil
import uuid
from typing import Optional, AsyncGenerator, Callable, Coroutine, Any, Dict, Set, TypeVar

T = TypeVar("T")

class WorktreeManagerV10:
    """
    V10 Git Worktree Manager for isolated subagent execution.
    Provides robust multi-agent task execution by preventing context bleeding
    and directory collisions through complete git worktree isolation.
    """

    def __init__(self, base_dir: str = "/tmp/magda_worktrees_v10") -> None:
        """
        Initializes the WorktreeManagerV10.

        Args:
            base_dir (str): Base directory under which worktree checkouts will be created.
        """
        self.base_dir: str = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.active_worktrees: Dict[str, str] = {}

    async def create_worktree(self, agent_id: Optional[str] = None, branch_name: Optional[str] = None) -> str:
        """
        Asynchronously creates an isolated git worktree.

        Args:
            agent_id (Optional[str]): A unique identifier for the agent or task.
            branch_name (Optional[str]): Optional branch name to create/checkout.

        Returns:
            str: The absolute path to the newly created worktree.

        Raises:
            RuntimeError: If worktree creation fails.
        """
        id_to_use: str = agent_id or str(uuid.uuid4())[:8]
        unique_suffix: str = str(uuid.uuid4())[:8]
        worktree_path: str = os.path.join(self.base_dir, f"worktree_{id_to_use}_{unique_suffix}")

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
                err_msg: str = stderr.decode().strip()
                logging.error(f"Failed to create git worktree: {err_msg}")
                raise RuntimeError(f"Git worktree creation failed: {err_msg}")

            logging.info(f"Created git worktree at {worktree_path}")
            self.active_worktrees[worktree_path] = id_to_use
            return worktree_path
        except Exception as e:
            logging.error(f"Error executing git worktree add: {e}")
            raise

    async def remove_worktree(self, worktree_path: str) -> None:
        """
        Asynchronously removes a git worktree and cleans up its directory.
        Forces removal if there are untracked or modified files.

        Args:
            worktree_path (str): The path to the worktree to remove.
        """
        if worktree_path not in self.active_worktrees:
            logging.warning(f"Path {worktree_path} not found in active worktrees, attempting removal anyway.")

        cmd = ["git", "worktree", "remove", "--force", worktree_path]
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logging.warning(f"Failed to cleanly remove git worktree via command: {stderr.decode().strip()}")
        except Exception as e:
            logging.error(f"Error removing git worktree at {worktree_path}: {e}")
        finally:
            if os.path.exists(worktree_path):
                try:
                    shutil.rmtree(worktree_path)
                except Exception as ex:
                    logging.error(f"Could not delete worktree directory {worktree_path}: {ex}")
            self.active_worktrees.pop(worktree_path, None)

    @asynccontextmanager
    async def isolated_environment(self, agent_id: Optional[str] = None, branch_name: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        An async context manager to wrap subtask execution with git worktree isolation.

        Args:
            agent_id (Optional[str]): A unique identifier for the agent or task.
            branch_name (Optional[str]): Optional branch name to use.

        Yields:
            str: Path to the isolated worktree directory.
        """
        worktree_path: str = await self.create_worktree(agent_id=agent_id, branch_name=branch_name)
        try:
            yield worktree_path
        finally:
            await self.remove_worktree(worktree_path)

    async def execute_in_isolation(
        self,
        task_coro_func: Callable[[str], Coroutine[Any, Any, T]],
        agent_id: Optional[str] = None,
        branch_name: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> T:
        """
        Executes an asynchronous callable inside an isolated Git worktree.
        Guarantees that the worktree is cleaned up even on failure, exception, or timeout.

        Args:
            task_coro_func (Callable): Async callable accepting worktree path (str) as argument.
            agent_id (Optional[str]): A unique identifier for the agent or task.
            branch_name (Optional[str]): Optional branch name to use.
            timeout (Optional[float]): Optional float specifying maximum execution time in seconds.

        Returns:
            T: The result of task_coro_func.

        Raises:
            asyncio.TimeoutError: If execution exceeds the timeout limit.
            Exception: Re-raises any exception thrown by task_coro_func.
        """
        async with self.isolated_environment(agent_id=agent_id, branch_name=branch_name) as worktree_path:
            task: Coroutine[Any, Any, T] = task_coro_func(worktree_path)
            if timeout is not None:
                try:
                    return await asyncio.wait_for(task, timeout=timeout)
                except asyncio.TimeoutError:
                    logging.error(f"Task execution timed out after {timeout}s in {worktree_path}")
                    raise
            else:
                return await task

    def get_active_worktrees(self) -> Dict[str, str]:
        """
        Returns a dictionary of currently active worktrees and their mapped agent IDs.

        Returns:
            Dict[str, str]: Mapped active worktrees.
        """
        return dict(self.active_worktrees)

    def verify_sandbox(self, worktree_path: str) -> bool:
        """
        Verifies that the given worktree path exists, is under the base_dir,
        and is isolated from other active worktrees.

        Args:
            worktree_path (str): Path to check.

        Returns:
            bool: True if verified successfully, False otherwise.
        """
        try:
            real_base: str = os.path.realpath(self.base_dir)
            real_path: str = os.path.realpath(worktree_path)

            if not os.path.exists(real_path):
                logging.warning(f"Sandbox verification failed: path {worktree_path} does not exist.")
                return False

            if not real_path.startswith(real_base):
                logging.warning(f"Sandbox verification failed: path {worktree_path} is not within base directory {self.base_dir}.")
                return False

            return True
        except Exception as e:
            logging.error(f"Exception during sandbox verification for path {worktree_path}: {e}")
            return False

    async def cleanup_all(self) -> None:
        """
        Removes all active worktrees tracked by this manager.
        """
        paths: list[str] = list(self.active_worktrees.keys())
        for path in paths:
            await self.remove_worktree(path)
