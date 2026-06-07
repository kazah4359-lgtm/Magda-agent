import asyncio
import logging
from typing import List, Dict, Any

from magda_agent.agents.sub_agent import SubAgent
from magda_agent.llm_client import LLMClient

class TeamOrchestrator:
    """
    Orchestrates the execution of multiple parallel subagents for a set of tasks.
    """
    def __init__(self, llm: LLMClient):
        """
        Initializes the TeamOrchestrator.
        """
        self.llm = llm

    async def parallel_execute(self, tasks: List[Dict[str, Any]], base_context: str) -> List[str]:
        """
        Executes multiple tasks in parallel using isolated SubAgents.
        """
        logging.info(f"Orchestrating {len(tasks)} parallel subagent tasks.")

        async def run_task(task_spec: Dict[str, Any]) -> str:
            sub_agent = SubAgent(llm=self.llm, use_isolation=True)
            task_description = task_spec.get('description', 'Unknown task')
            return await sub_agent.execute(task=task_description, context=base_context)

        results = await asyncio.gather(*(run_task(task) for task in tasks))
        return list(results)
