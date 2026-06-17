import asyncio
import logging
from typing import List, Dict, Any

from magda_agent.agents.spawner import SubagentSpawner
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
        self.spawner = SubagentSpawner(llm=llm)

    async def parallel_execute(self, tasks: List[Dict[str, Any]], base_context: str) -> List[str]:
        """
        Executes multiple tasks in parallel using isolated SubAgents dynamically spawned.
        """
        logging.info(f"Orchestrating {len(tasks)} parallel subagent tasks via SubagentSpawner.")
        return await self.spawner.spawn_and_execute(tasks, base_context, use_isolation=True)
