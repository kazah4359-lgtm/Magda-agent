from contextlib import asynccontextmanager
import asyncio
import logging
import os
import shutil
import uuid
from typing import Optional

class GitWorktreeManager:
    """
    Manages isolated Git worktrees for SubAgents.
    Ensures parallel tasks run in separate file system contexts.
    """
    def __init__(self, base_dir: str = "/tmp/magda_worktrees"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    async def create_worktree_async(self, branch_name: Optional[str] = None) -> str:
        """
        Creates a new git worktree. If branch_name is not provided,
        a detached worktree is created based on the current HEAD.
        Returns the path to the newly created worktree.
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
                logging.error(f"Failed to create git worktree: {stderr.decode()}")
                raise RuntimeError(f"Git worktree creation failed: {stderr.decode()}")

            logging.info(f"Created git worktree at {worktree_path}")
            return worktree_path
        except Exception as e:
            logging.error(f"Error executing git worktree add: {e}")
            raise

    async def remove_worktree_async(self, worktree_path: str) -> None:
        """
        Removes a git worktree and deletes its directory.
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
                logging.error(f"Failed to remove git worktree: {stderr.decode()}")
                # Fallback to manual deletion if git worktree remove fails
                if os.path.exists(worktree_path):
                    shutil.rmtree(worktree_path)
            else:
                 logging.info(f"Removed git worktree at {worktree_path}")
        except Exception as e:
            logging.error(f"Error executing git worktree remove: {e}")
            if os.path.exists(worktree_path):
                shutil.rmtree(worktree_path)

    @asynccontextmanager
    async def isolated_environment(self, branch_name: Optional[str] = None):
        """
        Context manager for creating and automatically cleaning up an isolated git worktree.
        """
        worktree_path = await self.create_worktree_async(branch_name=branch_name)
        try:
            yield worktree_path
        finally:
            await self.remove_worktree_async(worktree_path)
