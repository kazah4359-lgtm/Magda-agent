import logging
from typing import Optional
from magda_agent.llm_client import LLMClient
from magda_agent.isolation.git_worktree import GitWorktreeManager

class SubAgent:
    """
    SubAgent for executing isolated tasks in a separate context.
    Inspired by Claude Agent Teams and Hermes sub-agents.
    """
    def __init__(self, llm: LLMClient, system_prompt: Optional[str] = None, use_isolation: bool = False):
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

        worktree_path = None
        current_context = context

        if self.use_isolation and self.worktree_manager:
            try:
                worktree_path = await self.worktree_manager.create_worktree_async()
                current_context += f"\n\nIsolated Git Worktree Path: {worktree_path}"
                logging.info(f"SubAgent operating in isolated worktree: {worktree_path}")
            except Exception as e:
                logging.error(f"Failed to create isolated worktree: {e}")
                return f"Error: Failed to create isolated worktree - {e}"

        full_context = f"Parent Context:\n{current_context}\n\nAssigned Task:\n{task}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": full_context}
        ]

        try:
            result = await self.llm.chat_completion(messages)
            logging.info("SubAgent completed task.")
            return result
        except Exception as e:
            logging.error(f"Error executing SubAgent task: {e}")
            return f"Error executing SubAgent task: {e}"
        finally:
            if worktree_path and self.worktree_manager:
                try:
                    await self.worktree_manager.remove_worktree_async(worktree_path)
                except Exception as cleanup_error:
                    logging.error(f"Failed to cleanup worktree {worktree_path}: {cleanup_error}")