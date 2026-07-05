import asyncio
from typing import List, Dict, Optional

from magda_agent.llm_client import LLMClient
from magda_agent.agents.sub_agent import SubAgent


class MultiAgentManagerV3:
    """
    Manager for spawning and orchestrating multiple subagents concurrently.
    Inspired by Claude Agent Teams.
    """

    def __init__(self, llm: LLMClient):
        """
        Initializes the MultiAgentManagerV3.

        Args:
            llm (LLMClient): The LLMClient instance to provide to subagents.
        """
        self.llm = llm

    def spawn_subagent(self, system_prompt: Optional[str] = None, use_isolation: bool = False) -> SubAgent:
        """
        Spawns a single subagent with the given system prompt.

        Args:
            system_prompt (Optional[str]): The system prompt to configure the subagent with.
            use_isolation (bool): Whether the subagent should run in an isolated environment.

        Returns:
            SubAgent: The configured SubAgent instance.
        """
        return SubAgent(llm=self.llm, system_prompt=system_prompt, use_isolation=use_isolation)

    async def run_parallel_tasks(self, tasks: List[Dict[str, str]], context: str, use_isolation: bool = False) -> List[str]:
        """
        Runs multiple tasks in parallel using spawned subagents.

        Args:
            tasks (List[Dict[str, str]]): A list of dictionaries, where each dictionary
                                          contains 'task' (required) and 'system_prompt' (optional).
            context (str): The shared parent context to provide to all subagents.
            use_isolation (bool): Whether the subagents should run in isolated environments.

        Returns:
            List[str]: A list of results from each executed subagent task.
        """
        coroutines = []
        for task_info in tasks:
            task_desc = task_info.get('task', '')
            system_prompt = task_info.get('system_prompt')
            subagent = self.spawn_subagent(system_prompt=system_prompt, use_isolation=use_isolation)
            coroutines.append(subagent.execute(task=task_desc, context=context))

        results = await asyncio.gather(*coroutines)
        return list(results)
