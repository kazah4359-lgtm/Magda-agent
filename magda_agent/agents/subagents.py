import asyncio
import logging
from typing import List, Dict, Any, Optional

from magda_agent.agents.sub_agent import SubAgent
from magda_agent.llm_client import LLMClient

class SubagentsManager:
    """
    SubagentsManager coordinates and executes parallel tasks using isolated SubAgents.
    Inspired by Claude Agent SDK trends: subagent spawning for parallel execution
    using git worktree isolation.
    """
    def __init__(self, llm: LLMClient):
        """
        Initializes the SubagentsManager.

        Args:
            llm: The Language Model client to be used by spawned SubAgents.
        """
        self.llm = llm

    async def spawn_and_execute_parallel(self, tasks: List[Dict[str, Any]], context: str) -> List[str]:
        """
        Spawns subagents dynamically to execute multiple tasks concurrently.
        Enforces use_isolation=True to ensure Git worktree isolation.

        Args:
            tasks: A list of task dictionaries. Each dictionary should contain a 'description' field.
            context: The shared context passed to all subagents.

        Returns:
            A list of execution results corresponding to the order of tasks.
        """
        logging.info(f"SubagentsManager spawning {len(tasks)} isolated subagents for parallel execution.")

        async def run_subagent(task_spec: Dict[str, Any]) -> str:
            system_prompt: Optional[str] = task_spec.get('system_prompt', None)
            # Spawn the SubAgent enforcing worktree isolation.
            sub_agent = SubAgent(llm=self.llm, system_prompt=system_prompt, use_isolation=True)
            task_description = task_spec.get('description', 'Unknown task')
            return await sub_agent.execute(task=task_description, context=context)

        # Run all subagents concurrently
        results = await asyncio.gather(*(run_subagent(task) for task in tasks), return_exceptions=False)
        return list(results)
