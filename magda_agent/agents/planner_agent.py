import logging
from typing import Optional, List, Dict, Any
from magda_agent.planning.planner import Planner

class PlannerAgent:
    """
    Agent responsible for generating execution plans.
    """
    def __init__(self, planner: Optional[Planner]):
        self.planner = planner

    async def plan(self, user_input: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generates a plan based on the user input.
        """
        if not self.planner:
            return []

        if not self.planner.get_current_plan():
            await self.planner.generate_plan(user_input, user_id=user_id)

        return self.planner.get_current_plan()
