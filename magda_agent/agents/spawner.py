import asyncio
import logging
from typing import List, Dict, Any, Optional

from magda_agent.agents.sub_agent import SubAgent
from magda_agent.llm_client import LLMClient

class SubagentSpawner:
    """
    SubagentSpawner dynamically spawns SubAgents for concurrent tasks.
    Inspired by Claude Agent SDK: Agent Teams and Subagent spawning.
    """
    def __init__(self, llm: LLMClient):
        """
        Initializes the SubagentSpawner.

        Args:
            llm: The Language Model client to be used by spawned SubAgents.
        """
        self.llm = llm

    async def spawn_and_execute(self, tasks: List[Dict[str, Any]], base_context: str, use_isolation: bool = True) -> List[str]:
        """
        Spawns subagents dynamically to execute multiple tasks concurrently.

        Args:
            tasks: A list of task dictionaries containing a 'description' field.
            base_context: The shared context passed to all subagents.
            use_isolation: Whether to use Git Worktree isolation for each subagent.

        Returns:
            A list of execution results corresponding to the tasks.
        """
        logging.info(f"SubagentSpawner spawning {len(tasks)} isolated subagents for parallel execution.")

        async def run_subagent(task_spec: Dict[str, Any]) -> str:
            system_prompt: Optional[str] = task_spec.get('system_prompt', None)
            sub_agent = SubAgent(llm=self.llm, system_prompt=system_prompt, use_isolation=use_isolation)
            task_description = task_spec.get('description', 'Unknown task')
            return await sub_agent.execute(task=task_description, context=base_context)

        results = await asyncio.gather(*(run_subagent(task) for task in tasks))
        return list(results)
