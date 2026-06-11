import asyncio
import logging
from typing import List, Dict, Any

from magda_agent.agents.sub_agent import SubAgent
from magda_agent.llm_client import LLMClient

class HierarchicalDelegator:
    """
    Delegates tasks to a team of isolated sub-agents executing in parallel.
    """
    def __init__(self, llm: LLMClient):
        """
        Initializes the HierarchicalDelegator.
        """
        self.llm = llm

    async def delegate_tasks(self, tasks: List[str], base_context: str) -> List[Dict[str, Any]]:
        """
        Delegates multiple tasks to isolated sub-agents concurrently.
        """
        logging.info(f"HierarchicalDelegator delegating {len(tasks)} tasks.")

        async def execute_isolated_task(task: str) -> Dict[str, Any]:
            sub_agent = SubAgent(llm=self.llm, use_isolation=True)
            try:
                result = await sub_agent.execute(task=task, context=base_context)
                return {"task": task, "status": "success", "result": result}
            except Exception as e:
                logging.error(f"Task delegation failed: {e}")
                return {"task": task, "status": "error", "error": str(e)}

        results = await asyncio.gather(*(execute_isolated_task(task) for task in tasks))
        return list(results)
