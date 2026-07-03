import asyncio
import logging
import os
import shutil
import uuid
from typing import Optional, Dict

class WorktreeManagerV2:
    """
    Git worktree manager for sub-agents to allow parallel isolated execution.
    Inspired by Claude Agent SDK (Agent Teams with git worktree isolation).
    """

    def __init__(self, base_dir: str = "/tmp/magda_worktree_v2") -> None:
        """
        Initialize the WorktreeManagerV2.

        Args:
            base_dir (str): Base directory where worktrees will be created.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.active_worktrees: Dict[str, str] = {}

    async def create_worktree(self, agent_id: str, branch_name: Optional[str] = None) -> str:
        """
        Creates an isolated git worktree for an agent.

        Args:
            agent_id (str): A unique identifier for the agent.
            branch_name (Optional[str]): A branch name to create for the agent, defaults to detached HEAD.

        Returns:
            str: Path to the newly created worktree.
        """
        unique_suffix = str(uuid.uuid4())[:8]
        env_path = os.path.join(self.base_dir, f"subagent_{agent_id}_{unique_suffix}")

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
                error_msg = stderr.decode().strip()
                logging.error(f"Failed to create git worktree: {error_msg}")
                raise RuntimeError(f"Git worktree creation failed: {error_msg}")

            logging.info(f"Agent {agent_id} worktree created at {env_path}")
            self.active_worktrees[agent_id] = env_path
            return env_path
        except Exception as e:
            logging.error(f"Error during worktree creation for {agent_id}: {e}")
            raise

    async def cleanup_worktree(self, agent_id: str) -> None:
        """
        Removes the git worktree associated with an agent.

        Args:
            agent_id (str): The unique identifier of the agent.
        """
        env_path = self.active_worktrees.get(agent_id)
        if not env_path:
            logging.warning(f"No active worktree found for agent {agent_id}")
            return

        cmd = ["git", "worktree", "remove", "--force", env_path]
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logging.error(f"Failed to cleanly remove git worktree for {agent_id}: {stderr.decode().strip()}")
            else:
                logging.info(f"Successfully removed worktree for {agent_id}")
        except Exception as e:
            logging.error(f"Error removing worktree for {agent_id}: {e}")
        finally:
            if os.path.exists(env_path):
                try:
                    shutil.rmtree(env_path)
                except Exception as ex:
                    logging.error(f"Could not delete worktree folder for {agent_id}: {ex}")
            self.active_worktrees.pop(agent_id, None)

    async def cleanup_all(self) -> None:
        """
        Removes all active worktrees.
        """
        agents_to_cleanup = list(self.active_worktrees.keys())
        for agent_id in agents_to_cleanup:
            await self.cleanup_worktree(agent_id)
