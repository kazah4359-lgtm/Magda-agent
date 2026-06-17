import asyncio
import logging
import os
import shutil
import uuid
from typing import Optional

class IsolationManager:
    """
    Manages git worktree isolation for Agent Teams.
    """
    def __init__(self, base_dir: str = "/tmp/magda_team_worktrees") -> None:
        """
        Initializes the isolation manager.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    async def create_isolated_environment(self, branch_name: Optional[str] = None) -> str:
        """
        Creates an isolated git worktree environment.
        """
        env_id = str(uuid.uuid4())[:8]
        env_path = os.path.join(self.base_dir, f"env_{env_id}")

        if branch_name:
            cmd = ["git", "worktree", "add", "-b", branch_name, env_path, "HEAD"]
        else:
            cmd = ["git", "worktree", "add", "-d", env_path, "HEAD"]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                error_msg = stderr.decode()
                logging.error(f"Failed to create isolated environment: {error_msg}")
                raise RuntimeError(f"Git worktree creation failed: {error_msg}")

            logging.info(f"Created isolated environment at {env_path}")
            return env_path
        except Exception as e:
            logging.error(f"Error executing git worktree add: {e}")
            raise

    async def teardown_isolated_environment(self, env_path: str) -> None:
        """
        Tears down and removes an isolated git worktree environment.
        """
        cmd = ["git", "worktree", "remove", "--force", env_path]
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logging.error(f"Failed to remove isolated environment: {stderr.decode()}")
                if os.path.exists(env_path):
                    shutil.rmtree(env_path)
            else:
                logging.info(f"Removed isolated environment at {env_path}")
        except Exception as e:
            logging.error(f"Error executing git worktree remove: {e}")
            if os.path.exists(env_path):
                shutil.rmtree(env_path)
