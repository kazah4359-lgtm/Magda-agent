import logging
from typing import Optional
from magda_agent.llm_client import LLMClient
from magda_agent.isolation.git_worktree import GitWorktreeManager

class SubAgent:
    """
    SubAgent for executing isolated tasks in a separate context.
    Inspired by Claude Agent Teams.
    """
    def __init__(self, llm: LLMClient, system_prompt: Optional[str] = None, use_isolation: bool = False) -> None:
        """
        Initializes the SubAgent.
        """
        self.llm = llm
        self.system_prompt = system_prompt or "You are an isolated Sub-Agent executing a specific task."
        self.use_isolation = use_isolation
        self.worktree_manager = GitWorktreeManager() if use_isolation else None

    async def execute(self, task: str, context: str) -> str:
        """
        Executes a task given the context.
        """
        logging.info(f"SubAgent starting task: {task[:50]}...")

        current_context = context

        if self.use_isolation and self.worktree_manager:
            try:
                async with self.worktree_manager.isolated_environment() as worktree_path:
                    current_context += f"\n\nIsolated Git Worktree Path: {worktree_path}"
                    logging.info(f"SubAgent operating in isolated worktree: {worktree_path}")

                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Context:\n{current_context}\n\nTask:\n{task}"}
                    ]

                    result = await self.llm.chat_completion(messages)
                    logging.info("SubAgent completed task.")
                    return result
            except Exception as e:
                logging.error(f"Error during isolated execution: {e}")
                return f"Error: Failed during isolated execution - {e}"

        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{current_context}\n\nTask:\n{task}"}
            ]

            try:
                result = await self.llm.chat_completion(messages)
                logging.info("SubAgent completed task.")
                return result
            except Exception as e:
                logging.error(f"Error executing SubAgent task: {e}")
                return f"Error executing SubAgent task: {e}"
