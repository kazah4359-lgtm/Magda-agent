import logging
from typing import Optional, List, Dict, Any
from magda_agent.planning.planner import Planner
from magda_agent.emotions.mental_states import MentalState

class PlannerAgent:
    """
    Agent responsible for generating execution plans.
    """
    def __init__(self, planner: Optional[Planner], a2a_delegator=None):
        self.planner = planner
        self.a2a_delegator = a2a_delegator



    async def plan(
        self,
        user_input: str,
        user_id: Optional[str] = None,
        mental_state: Optional[MentalState] = None,
        behavior_weights: Optional[Dict[str, float]] = None,
        skill_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates a plan based on the user input.
        If a plan step requires a capability available on an external agent, it delegates the sub-plan.
        """
        if not self.planner:
            return []

        current_plan = self.planner.get_current_plan(user_id=user_id)
        if not current_plan:
            await self.planner.generate_plan(
                user_input,
                user_id=user_id,
                mental_state=mental_state,
                behavior_weights=behavior_weights,
                skill_weights=skill_weights
            )
            current_plan = self.planner.get_current_plan(user_id=user_id)

        # Retrieve plan and automatically delegate steps if an A2A Delegator is available
        if self.a2a_delegator and current_plan:
            for step in current_plan:
                if step.get("skill") == "delegate_to_agent":
                    capability = step.get("skill_kwargs", {}).get("capability")
                    if capability:
                        result = await self.delegate_subplan(capability, step)
                        step["result"] = result
                        logging.info(f"Step {step.get('id')} delegated successfully: {result}")

        return current_plan

    async def delegate_subplan(self, capability: str, plan_context: Dict[str, Any]) -> str:
        """
        Delegates a sub-plan to an external agent using the A2ADelegator.
        """
        if not self.a2a_delegator:
            return "Delegation failed: No A2ADelegator configured."
        return await self.a2a_delegator.delegate_subplan(capability, plan_context)
