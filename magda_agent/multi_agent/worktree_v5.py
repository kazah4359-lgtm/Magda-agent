import asyncio
import logging
import os
import shutil
import uuid
from typing import Optional, Dict
from contextlib import asynccontextmanager

class WorktreeManagerV5:
    """
    Manages isolated Git worktrees for sub-agents (v5).
    Ensures parallel tasks run in separate file system contexts to prevent cross-contamination.
    Inspired by Claude Agent SDK Agent Teams trend.
    """

    def __init__(self, base_dir: str = "/tmp/magda_worktree_v5") -> None:
        """
        Initialize the WorktreeManagerV5.

        Args:
            base_dir (str): Base directory where worktrees will be created.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.active_worktrees: Dict[str, str] = {}

    async def create_worktree(self, agent_id: Optional[str] = None, branch_name: Optional[str] = None) -> str:
        """
        Creates an isolated git worktree.

        Args:
            agent_id (Optional[str]): A unique identifier for the agent/task. If None, one is generated.
            branch_name (Optional[str]): A branch name to create, defaults to detached HEAD.

        Returns:
            str: Path to the newly created worktree.
        """
        id_to_use = agent_id or str(uuid.uuid4())[:8]
        unique_suffix = str(uuid.uuid4())[:8]
        worktree_path = os.path.join(self.base_dir, f"worktree_{id_to_use}_{unique_suffix}")

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
                error_msg = stderr.decode().strip()
                logging.error(f"Failed to create git worktree: {error_msg}")
                raise RuntimeError(f"Git worktree creation failed: {error_msg}")

            logging.info(f"Created git worktree at {worktree_path}")
            self.active_worktrees[worktree_path] = id_to_use
            return worktree_path
        except Exception as e:
            logging.error(f"Error during git worktree creation: {e}")
            raise

    async def remove_worktree(self, worktree_path: str) -> None:
        """
        Removes the git worktree and deletes its directory.

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
                logging.error(f"Failed to cleanly remove git worktree at {worktree_path}: {stderr.decode().strip()}")
            else:
                logging.info(f"Successfully removed git worktree at {worktree_path}")
        except Exception as e:
            logging.error(f"Error removing git worktree at {worktree_path}: {e}")
        finally:
            if os.path.exists(worktree_path):
                try:
                    shutil.rmtree(worktree_path)
                except Exception as ex:
                    logging.error(f"Could not delete worktree folder {worktree_path}: {ex}")
            self.active_worktrees.pop(worktree_path, None)

    @asynccontextmanager
    async def isolated_environment(self, agent_id: Optional[str] = None, branch_name: Optional[str] = None):
        """
        Asynchronous context manager for isolated environment.
        """
        worktree_path = await self.create_worktree(agent_id=agent_id, branch_name=branch_name)
        try:
            yield worktree_path
        finally:
            await self.remove_worktree(worktree_path)

    async def cleanup_all(self) -> None:
        """
        Removes all active worktrees tracked by this manager.
        """
        paths = list(self.active_worktrees.keys())
        for path in paths:
            await self.remove_worktree(path)
