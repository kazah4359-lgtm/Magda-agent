import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional

from magda_agent.llm_client import LLMClient
from magda_agent.agents.sub_agent import SubAgent
from magda_agent.architecture.agent_teams_v3 import AgentWorktreeIsolationV3

class ParallelSubagentSpawnerV5:
    """
    Manages spawning parallel subagents, utilizing Git worktree isolation
    to prevent cross-contamination between tasks. Inspired by Claude Agent Teams.
    """

    def __init__(self, llm: LLMClient, isolation_manager: Optional[AgentWorktreeIsolationV3] = None):
        """
        Initialize the ParallelSubagentSpawnerV5.

        Args:
            llm: The Language Model client for subagents.
            isolation_manager: Manager for Git worktree isolation. If not provided, a new one is created.
        """
        self.llm = llm
        self.isolation_manager = isolation_manager or AgentWorktreeIsolationV3()

    async def run_parallel_tasks(self, tasks: List[Dict[str, str]], base_context: str, use_isolation: bool = True) -> List[str]:
        """
        Run multiple tasks concurrently using spawned subagents, with optional isolated Git worktrees.

        Args:
            tasks: A list of dictionaries, each containing 'description' and optionally 'system_prompt'.
            base_context: The shared context string.
            use_isolation: If True, each subagent gets a dedicated Git worktree.

        Returns:
            A list containing the results of each task execution.
        """
        logging.info(f"ParallelSubagentSpawnerV5 spawning {len(tasks)} tasks (isolation={use_isolation}).")

        async def execute_isolated_task(task_info: Dict[str, str]) -> str:
            agent_id = str(uuid.uuid4())[:8]
            task_desc = task_info.get("description", "")
            system_prompt = task_info.get("system_prompt", "You are an isolated Sub-Agent executing a task.")

            subagent = SubAgent(llm=self.llm, system_prompt=system_prompt, use_isolation=False) # We handle isolation here externally
            worktree_path = None
            task_context = base_context

            if use_isolation:
                try:
                    worktree_path = await self.isolation_manager.create_worktree(agent_id=agent_id)
                    task_context += f"\n\nIsolated Git Worktree Path: {worktree_path}"
                    logging.info(f"Agent {agent_id} using worktree {worktree_path}")
                except Exception as e:
                    logging.error(f"Failed to create worktree for agent {agent_id}: {e}")
                    return f"Error: Failed to create isolated worktree - {e}"

            try:
                result = await subagent.execute(task=task_desc, context=task_context)
                return result
            except Exception as e:
                logging.error(f"Task execution failed for agent {agent_id}: {e}")
                return f"Error: Task execution failed - {e}"
            finally:
                if use_isolation and worktree_path:
                    try:
                        await self.isolation_manager.remove_worktree(agent_id=agent_id)
                    except Exception as e:
                        logging.error(f"Failed to clean up worktree for agent {agent_id}: {e}")

        coroutines = [execute_isolated_task(task) for task in tasks]
        results = await asyncio.gather(*coroutines)
        return list(results)
