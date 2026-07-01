from typing import Dict, Any, List
import logging
import asyncio
from magda_agent.integration.a2a_discovery import A2ADiscovery
from magda_agent.integration.a2a_delegation import A2ADelegator

class A2AOrchestrator:
    """
    Coordinates dispatching tasks to multiple A2A sub-agents using A2ADelegator.
    Supports concurrent delegation for parallelizable tasks.
    """
    def __init__(self, discovery: A2ADiscovery, delegator: A2ADelegator):
        """
        Initializes the orchestrator with discovery and delegator components.
        """
        self.discovery = discovery
        self.delegator = delegator

    async def dispatch_concurrently(self, sub_plans: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Dispatches multiple sub-plans concurrently to peer agents.

        Args:
            sub_plans: A list of sub-plans (where each sub-plan is a dict containing capability and steps).

        Returns:
            A dictionary mapping step IDs to their delegation result.
        """
        results = {}
        tasks = []

        async def _delegate_and_record(capability: str, step: Dict[str, Any]):
            step_id = step.get("id")
            result = await self.delegator.delegate_subplan(capability, step)
            return step_id, result

        for sub_plan in sub_plans:
            capability = sub_plan.get("capability")
            for step in sub_plan.get("steps", []):
                tasks.append(_delegate_and_record(capability, step))

        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in enumerate(completed_tasks):
            if isinstance(result, Exception):
                logging.error(f"Error during concurrent delegation: {result}")
                # We can't easily get the step_id here if the exception happened outside of our inner wrapper,
                # but because we wrapped it, we should get (step_id, exception/result) unless the wrapper itself failed.
            else:
                step_id, delegation_result = result
                if step_id:
                    results[step_id] = delegation_result

        return results

    async def execute_orchestrated_plan(self, plan: List[Dict[str, Any]], concurrent: bool = False) -> Dict[str, str]:
        """
        Executes a full plan by splitting it into sub-plans and dispatching them either concurrently or sequentially.

        Args:
            plan: The full execution plan.
            concurrent: If True, dispatches sub-plans concurrently. If False, executes them sequentially.

        Returns:
            A dictionary mapping step IDs to their delegation result.
        """
        if not plan:
            return {}

        sub_plans = self.delegator.split_plan(plan)

        if concurrent:
            return await self.dispatch_concurrently(sub_plans)
        else:
            return await self.delegator.execute_plan(plan)
