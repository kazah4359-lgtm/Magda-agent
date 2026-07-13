import asyncio
import logging
from typing import List, Dict, Any

from magda_agent.llm_client import LLMClient
from magda_agent.agents.sub_agent import SubAgent
from magda_agent.isolation.git_worktree import GitWorktreeManager

class WorktreeDispatcherV6:
    """
    WorktreeDispatcherV6 coordinates a team of SubAgents, ensuring each
    operates within an isolated Git worktree for parallel task execution.
    Inspired by Claude Agent Teams trend (June 2026).
    """
    def __init__(self, llm: LLMClient, base_dir: str = "/tmp/magda_worktrees_v6"):
        """
        Initializes the WorktreeDispatcherV6.

        Args:
            llm: The LLM client for subagents.
            base_dir: Base directory for Git worktrees.
        """
        self.llm = llm
        self.worktree_manager = GitWorktreeManager(base_dir=base_dir)

    async def dispatch_tasks(self, tasks: List[Dict[str, Any]], base_context: str) -> List[str]:
        """
        Dispatches multiple tasks to parallel subagents with worktree isolation.

        Args:
            tasks: List of task specifications (description, system_prompt).
            base_context: Shared context for all subagents.

        Returns:
            List of execution results.
        """
        logging.info(f"WorktreeDispatcherV6 dispatching {len(tasks)} tasks with isolation.")

        async def run_task(task_spec: Dict[str, Any]) -> str:
            """
            Executes a single subagent task within an isolated worktree.

            Args:
                task_spec: The specification for the individual task.

            Returns:
                The result of the task execution.
            """
            task_desc = task_spec.get("description", "No description provided.")
            system_prompt = task_spec.get("system_prompt", "You are an isolated Sub-Agent.")

            try:
                # Use the isolated_environment context manager from GitWorktreeManager for robust cleanup
                async with self.worktree_manager.isolated_environment() as worktree_path:
                    logging.info(f"Subagent operating in isolated worktree: {worktree_path}")

                    # Create subagent without its own internal isolation since we manage it here
                    sub_agent = SubAgent(llm=self.llm, system_prompt=system_prompt, use_isolation=False)

                    # Augment context with worktree info
                    augmented_context = f"{base_context}\n\nIsolated Worktree Path: {worktree_path}"

                    result = await sub_agent.execute(task=task_desc, context=augmented_context)
                    return result
            except Exception as e:
                logging.error(f"Failed to dispatch subagent task: {e}")
                return f"Error: Dispatch failed - {e}"

        results = await asyncio.gather(*(run_task(task) for task in tasks))
        return list(results)
