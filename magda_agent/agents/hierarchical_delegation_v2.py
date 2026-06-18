import asyncio
import logging
from typing import List, Dict, Any, Optional

from magda_agent.llm_client import LLMClient
from magda_agent.agents.sub_agent import SubAgent
from magda_agent.isolation.git_worktree import GitWorktreeManager

class HierarchicalDelegatorV2:
    """
    HierarchicalDelegatorV2 is responsible for recursively delegating tasks
    to specialized sub-agents using strict Git worktree isolation.
    """
    def __init__(self, llm: LLMClient, base_worktree_dir: str = "/tmp/magda_hierarchical_worktrees") -> None:
        """
        Initializes the HierarchicalDelegatorV2.

        Args:
            llm (LLMClient): The LLM client to be used by all spawned sub-agents.
            base_worktree_dir (str): Base directory for creating isolated Git worktrees.
        """
        self.llm = llm
        self.worktree_manager = GitWorktreeManager(base_dir=base_worktree_dir)

    async def delegate_task(self, task: Dict[str, Any], context: str, depth: int = 0, max_depth: int = 3) -> str:
        """
        Delegates a single task. If the task has sub-tasks, it recursively delegates them.

        Args:
            task (Dict[str, Any]): A dictionary containing task specifications. It may contain a 'sub_tasks' key.
            context (str): The common context for the task.
            depth (int): Current depth of delegation hierarchy.
            max_depth (int): Maximum allowed depth for recursive delegation.

        Returns:
            str: The result of the task execution.
        """
        task_description = task.get('description', 'Unknown task')
        logging.info(f"Delegating task at depth {depth}: {task_description[:50]}...")

        # Base case: max depth reached
        if depth >= max_depth:
            logging.warning(f"Max delegation depth {max_depth} reached. Executing immediately.")
            return await self._execute_leaf_task(task_description, context)

        sub_tasks: Optional[List[Dict[str, Any]]] = task.get('sub_tasks')

        # If there are sub_tasks, delegate them concurrently
        if sub_tasks and len(sub_tasks) > 0:
            logging.info(f"Task has {len(sub_tasks)} sub-tasks. Delegating recursively...")
            # Recursive delegation
            sub_results = await asyncio.gather(*(
                self.delegate_task(sub_task, context, depth + 1, max_depth)
                for sub_task in sub_tasks
            ))

            # Combine sub-results to form a new context for a final summarization/execution step
            combined_sub_results = "\n\n".join([f"Sub-task Result {i+1}:\n{res}" for i, res in enumerate(sub_results)])
            enriched_context = f"{context}\n\nSub-tasks results:\n{combined_sub_results}"

            logging.info("Sub-tasks completed. Executing parent task with enriched context.")
            return await self._execute_leaf_task(task_description, enriched_context)

        else:
            # Leaf node: execute directly in isolated context
            return await self._execute_leaf_task(task_description, context)

    async def _execute_leaf_task(self, description: str, context: str) -> str:
        """
        Executes a single leaf task using an isolated SubAgent.

        Args:
            description (str): Description of the task to be executed.
            context (str): Context for the execution.

        Returns:
            str: The execution result.
        """
        worktree_path = None
        try:
            # Create isolated environment
            worktree_path = await self.worktree_manager.create_worktree_async()
            enriched_context = f"{context}\n\nOperating in isolated Worktree Path: {worktree_path}"

            # SubAgent uses its own LLMClient and context compressor
            sub_agent = SubAgent(llm=self.llm, use_isolation=False) # Isolation handled explicitly here via context

            result = await sub_agent.execute(task=description, context=enriched_context)
            return result

        except Exception as e:
            logging.error(f"Error executing leaf task: {e}")
            return f"Task Failed: {str(e)}"
        finally:
            if worktree_path:
                try:
                    await self.worktree_manager.remove_worktree_async(worktree_path)
                except Exception as cleanup_error:
                    logging.error(f"Failed to cleanup worktree {worktree_path}: {cleanup_error}")
