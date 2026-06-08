import asyncio
import logging
from typing import List, Dict, Any

from magda_agent.agents.sub_agent import SubAgent
from magda_agent.llm_client import LLMClient

class TeamManager:
    """
    Manages the execution of multiple sub-agents in parallel using git worktree isolation.
    """
    def __init__(self, llm: LLMClient):
        """
        Initializes the TeamManager.
        """
        self.llm = llm

    async def spawn_and_execute(self, tasks: List[Dict[str, Any]], context: str) -> List[str]:
        """
        Executes a list of tasks concurrently by spawning isolated SubAgents.
        """
        logging.info(f"TeamManager spawning {len(tasks)} isolated sub-agents.")

        async def run_task(task_spec: Dict[str, Any]) -> str:
            sub_agent = SubAgent(llm=self.llm, use_isolation=True)
            task_desc = task_spec.get('description', 'Unknown task')
            return await sub_agent.execute(task=task_desc, context=context)

        results = await asyncio.gather(*(run_task(task) for task in tasks))
        return list(results)
