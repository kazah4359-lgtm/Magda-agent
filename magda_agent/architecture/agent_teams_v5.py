import asyncio
import logging
import os
import shutil
import uuid
import time
from typing import Optional, List, Dict, Tuple, Any


class GitWorktreeError(Exception):
    """Exception raised for errors during git worktree operations."""
    pass


class AgentWorktreeIsolationV5:
    """
    Manages isolated git worktrees for individual sub-agents with aggressive cleanup and synchronization.
    Provides isolation logic so multiple sub-agents can work without cross-contamination.
    """

    def __init__(self, base_dir: str = "/tmp/magda_agent_teams_v5") -> None:
        """
        Initialize the isolation manager.

        Args:
            base_dir (str): Base directory where worktrees will be created.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.active_worktrees: Dict[str, str] = {}
        self.worktree_branches: Dict[str, str] = {}

    async def create_worktree(self, agent_id: str, branch_name: Optional[str] = None) -> Tuple[str, Dict[str, str]]:
        """
        Creates an isolated git worktree for an agent and returns isolated environment variables.

        Args:
            agent_id (str): A unique identifier for the agent.
            branch_name (Optional[str]): A branch name to create for the agent, defaults to auto-generated branch.

        Returns:
            Tuple[str, Dict[str, str]]: Path to the newly created worktree and its isolated environment variables.
        """
        unique_suffix = str(uuid.uuid4())[:8]
        env_path = os.path.join(self.base_dir, f"agent_{agent_id}_{unique_suffix}")
        effective_branch = branch_name or f"subagent/{agent_id}_{unique_suffix}"

        cmd = ["git", "worktree", "add", "-b", effective_branch, env_path, "HEAD"]

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
                raise GitWorktreeError(f"Git worktree creation failed: {error_msg}")

            logging.info(f"Agent {agent_id} worktree created at {env_path} on branch {effective_branch}")
            self.active_worktrees[agent_id] = env_path
            self.worktree_branches[agent_id] = effective_branch

            # Create isolated environment variables
            isolated_env = os.environ.copy()
            isolated_env["MAGDA_AGENT_ID"] = agent_id
            isolated_env["MAGDA_WORKTREE_PATH"] = env_path
            isolated_env["MAGDA_WORKTREE_BRANCH"] = effective_branch
            isolated_env["MAGDA_ISOLATED"] = "true"

            return env_path, isolated_env
        except Exception as e:
            logging.error(f"Error during worktree creation for {agent_id}: {e}")
            raise

    async def synchronize_worktree(
        self,
        agent_id: str,
        target_branch: str = "main",
        commit_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronizes changes from a sub-agent's worktree back to the central repository branch.

        Args:
            agent_id (str): The identifier of the sub-agent.
            target_branch (str): The target central branch to synchronize changes into.
            commit_message (Optional[str]): Commit message for changes in the worktree.

        Returns:
            Dict[str, Any]: Synchronization summary including status and commit details.
        """
        env_path = self.active_worktrees.get(agent_id)
        if not env_path or not os.path.exists(env_path):
            raise GitWorktreeError(f"No active worktree path found for agent {agent_id}")

        msg = commit_message or f"Sync sub-agent {agent_id} worktree changes"
        branch = self.worktree_branches.get(agent_id)

        # 1. Stage changes in worktree
        add_proc = await asyncio.create_subprocess_exec(
            "git", "add", "-A",
            cwd=env_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, add_err = await add_proc.communicate()
        if add_proc.returncode != 0:
            raise GitWorktreeError(f"Failed to stage changes in worktree for {agent_id}: {add_err.decode().strip()}")

        # 2. Check status / Commit changes
        status_proc = await asyncio.create_subprocess_exec(
            "git", "status", "--porcelain",
            cwd=env_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        status_out, _ = await status_proc.communicate()

        committed = False
        if status_out.strip():
            commit_proc = await asyncio.create_subprocess_exec(
                "git", "commit", "-m", msg,
                cwd=env_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, commit_err = await commit_proc.communicate()
            if commit_proc.returncode != 0:
                raise GitWorktreeError(f"Failed to commit changes in worktree for {agent_id}: {commit_err.decode().strip()}")
            committed = True

        # 3. Merge or merge-commit branch back to main repo / target_branch
        # Run git merge in main repository (root directory or main worktree parent)
        main_repo_dir = os.getcwd()
        if branch:
            merge_proc = await asyncio.create_subprocess_exec(
                "git", "merge", "--no-ff", branch, "-m", f"Merge worktree branch {branch} for agent {agent_id}",
                cwd=main_repo_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout_merge, stderr_merge = await merge_proc.communicate()
            if merge_proc.returncode != 0:
                err_msg = stderr_merge.decode().strip() or stdout_merge.decode().strip()
                raise GitWorktreeError(f"Failed to synchronize worktree branch {branch} into {target_branch}: {err_msg}")

        return {
            "success": True,
            "agent_id": agent_id,
            "worktree_path": env_path,
            "branch": branch,
            "target_branch": target_branch,
            "committed": committed,
            "synced": True
        }

    async def remove_worktree(self, agent_id: str) -> None:
        """
        Removes the git worktree associated with an agent with aggressive cleanup fallback.

        Args:
            agent_id (str): The unique identifier of the agent.
        """
        env_path = self.active_worktrees.get(agent_id)
        if not env_path:
            logging.warning(f"No active worktree found for agent {agent_id}")
            return

        cmd = ["git", "worktree", "remove", "--force", env_path]
        git_success = False
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
                git_success = True
        except Exception as e:
            logging.error(f"Error executing git worktree remove for {agent_id}: {e}")
        finally:
            if os.path.exists(env_path):
                logging.warning(f"Worktree path {env_path} still exists. Attempting aggressive cleanup.")
                await self._aggressive_cleanup(env_path)
            self.active_worktrees.pop(agent_id, None)
            self.worktree_branches.pop(agent_id, None)

    async def _aggressive_cleanup(self, env_path: str, retries: int = 3, delay: float = 1.0) -> None:
        """
        Aggressively removes a directory with retries.

        Args:
            env_path (str): The path to remove.
            retries (int): Number of retries.
            delay (float): Delay between retries.
        """
        for attempt in range(retries):
            try:
                shutil.rmtree(env_path, ignore_errors=True)
                if not os.path.exists(env_path):
                    logging.info(f"Aggressive cleanup succeeded for {env_path}")
                    return
            except Exception as ex:
                logging.error(f"Attempt {attempt + 1}: Aggressive cleanup failed for {env_path}: {ex}")

            if attempt < retries - 1:
                await asyncio.sleep(delay)

        logging.error(f"Aggressive cleanup failed for {env_path} after {retries} attempts.")


class AgentTeamManagerV5:
    """
    Coordinates a team of agents operating in isolated worktrees with worktree synchronization and aggressive cleanup.
    """

    def __init__(self, isolation_manager: Optional[AgentWorktreeIsolationV5] = None) -> None:
        """
        Initialize the Agent Team Manager V5.

        Args:
            isolation_manager (Optional[AgentWorktreeIsolationV5]): Worktree isolation manager to use.
        """
        self.isolation_manager = isolation_manager or AgentWorktreeIsolationV5()
        self.agents: List[str] = []
        self.agent_envs: Dict[str, Dict[str, str]] = {}

    async def spawn_agent(self, agent_id: str, branch_name: Optional[str] = None) -> Tuple[str, Dict[str, str]]:
        """
        Spawns a new agent and sets up its isolated worktree.

        Args:
            agent_id (str): A unique string identifying the agent.
            branch_name (Optional[str]): Branch for the worktree.

        Returns:
            Tuple[str, Dict[str, str]]: Path to the agent's worktree and its isolated environment.
        """
        if agent_id in self.agents:
            raise ValueError(f"Agent {agent_id} already exists.")

        worktree_path, isolated_env = await self.isolation_manager.create_worktree(agent_id, branch_name)
        self.agents.append(agent_id)
        self.agent_envs[agent_id] = isolated_env
        return worktree_path, isolated_env

    def get_agent_env(self, agent_id: str) -> Optional[Dict[str, str]]:
        """
        Retrieves the isolated environment for a spawned agent.

        Args:
            agent_id (str): A unique string identifying the agent.

        Returns:
            Optional[Dict[str, str]]: The isolated environment dictionary, if agent exists.
        """
        return self.agent_envs.get(agent_id)

    async def synchronize_subagent(
        self,
        agent_id: str,
        target_branch: str = "main",
        commit_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronizes sub-agent worktree changes into the main central repository branch.

        Args:
            agent_id (str): Unique agent identifier.
            target_branch (str): Target branch name.
            commit_message (Optional[str]): Optional commit message.

        Returns:
            Dict[str, Any]: Synchronization result dictionary.
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} is not registered in this team manager.")

        return await self.isolation_manager.synchronize_worktree(
            agent_id=agent_id,
            target_branch=target_branch,
            commit_message=commit_message
        )

    async def synchronize_worktree(
        self,
        agent_id: str,
        target_branch: str = "main",
        commit_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Alias for synchronize_subagent to maintain compatibility."""
        return await self.synchronize_subagent(agent_id, target_branch, commit_message)

    async def disband_agent(self, agent_id: str, sync_before_disband: bool = False) -> None:
        """
        Disbands an agent and aggressively cleans up its worktree.

        Args:
            agent_id (str): The identifier of the agent to disband.
            sync_before_disband (bool): If True, synchronizes changes before cleanup.
        """
        if agent_id not in self.agents:
            logging.warning(f"Cannot disband unknown agent {agent_id}")
            return

        if sync_before_disband:
            try:
                await self.synchronize_subagent(agent_id)
            except Exception as ex:
                logging.error(f"Synchronization failed prior to disbanding agent {agent_id}: {ex}")

        await self.isolation_manager.remove_worktree(agent_id)
        self.agents.remove(agent_id)
        self.agent_envs.pop(agent_id, None)

    async def disband_all(self, sync_before_disband: bool = False) -> None:
        """
        Disbands all active agents and aggressively cleans up their worktrees.

        Args:
            sync_before_disband (bool): If True, attempts synchronization before disbanding.
        """
        agents_to_disband = list(self.agents)
        for agent_id in agents_to_disband:
            await self.disband_agent(agent_id, sync_before_disband=sync_before_disband)


class AgentEvaluatorTeamV5:
    """
    Subagent evaluator pool V5 that can evaluate code and synchronize worktree results.
    """

    def __init__(self, evaluators: List[str], team_manager: Optional[AgentTeamManagerV5] = None) -> None:
        """
        Initialize the evaluator team V5.

        Args:
            evaluators (List[str]): The list of evaluator IDs.
            team_manager (Optional[AgentTeamManagerV5]): Manager for strict sandboxing.
        """
        self.evaluators = evaluators
        self.team_manager = team_manager or AgentTeamManagerV5()

    async def evaluate_code(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate generator code iteratively within strict sandboxes and synchronize worktrees if needed.

        Args:
            code (str): The code to evaluate.
            context (Dict[str, Any]): Additional context.

        Returns:
            Dict[str, Any]: The evaluation results.
        """
        results = []
        for evaluator in self.evaluators:
            try:
                worktree_path, isolated_env = await self.team_manager.spawn_agent(evaluator)
            except ValueError:
                isolated_env = self.team_manager.get_agent_env(evaluator) or {}

            try:
                result = await self._call_llm_evaluator(evaluator, code, context, isolated_env)
                results.append(result)
            finally:
                await self.team_manager.disband_agent(evaluator)

        passed = bool(results) and all(r.get("passed", False) for r in results)

        return {
            "passed": passed,
            "results": results,
            "overall_score": sum(r.get("score", 0) for r in results) / len(results) if results else 0.0
        }

    async def _call_llm_evaluator(self, evaluator_id: str, code: str, context: Dict[str, Any], isolated_env: Dict[str, str]) -> Dict[str, Any]:
        """
        Call the LLM for evaluation.
        """
        from magda_agent.llm_client import LLMClient
        client = LLMClient()
        prompt = f"Evaluate this code:\n{code}\nContext: {context}\nEvaluator ID: {evaluator_id}\nSandbox: {isolated_env.get('MAGDA_WORKTREE_PATH')}"

        response = await client.generate(prompt=prompt)

        return {
            "evaluator_id": evaluator_id,
            "passed": "PASSED" in response,
            "score": 100 if "PASSED" in response else 50,
            "feedback": response
        }
