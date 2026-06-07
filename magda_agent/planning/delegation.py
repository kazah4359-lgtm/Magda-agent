import logging
import asyncio
from typing import List, Dict, Any

class AgentDelegationProtocol:
    """
    Protocol for the Prefrontal Cortex (Planner) to identify parallelizable sub-plans
    and dispatch them to isolated sub-agents concurrently.
    """

    def __init__(self, planner: Any):
        """
        Initializes the delegation protocol with a reference to the Planner.

        Args:
            planner (Any): The Planner instance to delegate from.
        """
        self.planner = planner

    async def execute_delegated_steps(self, steps: List[Dict[str, Any]], context: str) -> Dict[str, str]:
        """
        Routes the given steps to sub-agents concurrently and awaits their results.

        Args:
            steps (List[Dict[str, Any]]): The steps flagged for delegation.
            context (str): The context to pass to the sub-agents.

        Returns:
            Dict[str, str]: A dictionary mapping step IDs to their results.
        """
        results = {}

        async def _execute_single_step(step: Dict[str, Any]) -> tuple[str, str]:
            """
            Executes a single step by spawning a sub-agent.

            Args:
                step (Dict[str, Any]): A dictionary containing step definition with id and description.

            Returns:
                tuple[str, str]: A tuple of step_id and result string.
            """
            step_id = step.get("id")
            description = step.get("description", "")
            if not step_id or not description:
                logging.warning(f"Skipping invalid step for delegation: {step}")
                return step_id or "unknown", "invalid"

            try:
                result = await self.planner.spawn_sub_agent(task=description, context=context)
                return step_id, result
            except Exception as e:
                logging.error(f"Error delegating step {step_id}: {e}")
                return step_id, f"Error: {e}"

        tasks = []
        for step in steps:
            tasks.append(asyncio.create_task(_execute_single_step(step)))

        if tasks:
            gathered_results = await asyncio.gather(*tasks)
            for step_id, res in gathered_results:
                if step_id != "unknown" and res != "invalid":
                    results[step_id] = res

        return results
