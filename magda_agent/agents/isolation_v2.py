import asyncio
import logging
import os
import shutil
import uuid
from typing import Optional, Dict

class GitWorktreeIsolationV2:
    """
    Manages strict git worktree isolation and context separation for subagents.
    Inspired by Claude Agent Teams trend.
    """
    def __init__(self, base_dir: str = "/tmp/magda_team_worktrees_v2") -> None:
        """
        Initializes the isolation manager.

        Args:
            base_dir: The base directory where worktrees will be created.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.active_contexts: Dict[str, str] = {}

    async def setup_isolation(self, agent_id: str, branch_name: Optional[str] = None) -> str:
        """
        Sets up an isolated git worktree environment for a subagent.

        Args:
            agent_id: A unique identifier for the subagent.
            branch_name: Optional branch name. If not provided, creates a detached worktree.

        Returns:
            The path to the isolated worktree environment.
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
                logging.error(f"Failed to create isolated environment for {agent_id}: {error_msg}")
                raise RuntimeError(f"Git worktree creation failed: {error_msg}")

            logging.info(f"Created isolated environment for {agent_id} at {env_path}")
            self.active_contexts[agent_id] = env_path
            return env_path
        except Exception as e:
            logging.error(f"Error executing git worktree add for {agent_id}: {e}")
            raise

    async def teardown_isolation(self, agent_id: str) -> None:
        """
        Tears down and removes an isolated git worktree environment.

        Args:
            agent_id: The unique identifier for the subagent.
        """
        env_path = self.active_contexts.get(agent_id)
        if not env_path:
            logging.warning(f"No active isolation context found for agent {agent_id}")
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
                logging.error(f"Failed to remove isolated environment for {agent_id}: {stderr.decode().strip()}")
            else:
                logging.info(f"Removed isolated environment for {agent_id} at {env_path}")
        except Exception as e:
            logging.error(f"Error executing git worktree remove for {agent_id}: {e}")
        finally:
            if os.path.exists(env_path):
                try:
                    shutil.rmtree(env_path)
                except Exception as ex:
                    logging.error(f"Could not delete worktree folder for {agent_id}: {ex}")
            self.active_contexts.pop(agent_id, None)
