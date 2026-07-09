import json
import logging
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field, ValidationError
from magda_agent.llm_client import LLMClient
from magda_agent.skills.registry import SkillRegistry
from magda_agent.learning.habits import HabitTracker
from magda_agent.agents.teams import TeamManager
from magda_agent.planning.dependency_graph import DependencyGraph
from magda_agent.emotions.mental_states import MentalState


class PlanStep(BaseModel):
    id: str = Field(default_factory=lambda: "")
    description: str
    skill: Optional[str] = None
    skill_kwargs: Optional[Dict[str, Any]] = None
    dependencies: List[str] = Field(default_factory=list)


class TypedPlan(BaseModel):
    goal: str
    constraints: List[str] = Field(default_factory=list)
    risk: str
    steps: List[PlanStep]
    acceptance: List[str] = Field(default_factory=list)


class UserPlanState(BaseModel):
    """Mutable planning state isolated per user/request owner."""

    current_plan: List[Dict[str, Any]] = Field(default_factory=list)
    completed_steps: List[Dict[str, Any]] = Field(default_factory=list)
    current_goal: Optional[str] = None
    current_constraints: List[str] = Field(default_factory=list)
    current_risk: Optional[str] = None
    current_acceptance: List[str] = Field(default_factory=list)
    paused_plan: Optional[Dict[str, Any]] = None


class Planner:
    """
    Prefrontal Cortex (Planner) module.
    Responsible for breaking down complex queries into steps (plans),
    selecting which skills to use, resolving dependencies (DAG), and maintaining state.
    """

    DEFAULT_STATE_KEY = "__default__"

    def __init__(self, llm: LLMClient, skills: SkillRegistry, habit_tracker: Optional[HabitTracker] = None):
        self.llm = llm
        self.skills = skills
        self.habit_tracker = habit_tracker
        self.user_states: Dict[str, UserPlanState] = {}

    def _state_key(self, user_id: Optional[Any] = None) -> str:
        return self.DEFAULT_STATE_KEY if user_id is None else str(user_id)

    def get_user_state(self, user_id: Optional[Any] = None) -> UserPlanState:
        key = self._state_key(user_id)
        if key not in self.user_states:
            self.user_states[key] = UserPlanState()
        return self.user_states[key]

    # Backwards-compatible properties for tests and single-user callers. New
    # concurrent code should pass user_id to the methods below.
    @property
    def current_plan(self) -> List[Dict[str, Any]]:
        return self.get_user_state().current_plan

    @current_plan.setter
    def current_plan(self, value: List[Dict[str, Any]]) -> None:
        self.get_user_state().current_plan = value

    @property
    def completed_steps(self) -> List[Dict[str, Any]]:
        return self.get_user_state().completed_steps

    @completed_steps.setter
    def completed_steps(self, value: List[Dict[str, Any]]) -> None:
        self.get_user_state().completed_steps = value

    @property
    def current_goal(self) -> Optional[str]:
        return self.get_user_state().current_goal

    @current_goal.setter
    def current_goal(self, value: Optional[str]) -> None:
        self.get_user_state().current_goal = value

    @property
    def current_constraints(self) -> List[str]:
        return self.get_user_state().current_constraints

    @current_constraints.setter
    def current_constraints(self, value: List[str]) -> None:
        self.get_user_state().current_constraints = value

    @property
    def current_risk(self) -> Optional[str]:
        return self.get_user_state().current_risk

    @current_risk.setter
    def current_risk(self, value: Optional[str]) -> None:
        self.get_user_state().current_risk = value

    @property
    def current_acceptance(self) -> List[str]:
        return self.get_user_state().current_acceptance

    @current_acceptance.setter
    def current_acceptance(self, value: List[str]) -> None:
        self.get_user_state().current_acceptance = value

    @property
    def paused_plan(self) -> Optional[Dict[str, Any]]:
        return self.get_user_state().paused_plan

    @paused_plan.setter
    def paused_plan(self, value: Optional[Dict[str, Any]]) -> None:
        self.get_user_state().paused_plan = value

    async def generate_plan(
        self,
        user_input: str,
        user_id: Optional[str] = None,
        mental_state: Optional[MentalState] = None,
        behavior_weights: Optional[Dict[str, float]] = None,
        skill_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        logging.info("Generating plan for input")
        state = self.get_user_state(user_id)
        skills_desc = self.skills.get_skills_summary()

        system_prompt = (
            "You are the Prefrontal Cortex of an AI agent. Your job is to break down "
            "the user's request into a logical sequence of steps.\n"
            "Available skills:\n"
            f"{skills_desc}\n"
            "Return a JSON object with the following keys:\n"
            "- 'goal': a string summarizing the objective\n"
            "- 'constraints': an array of string constraints\n"
            "- 'risk': a string indicating the risk level (e.g., low, medium, high)\n"
            "- 'steps': an array of step objects. Each step must have 'id' (string), 'description', 'skill' (or null), 'skill_kwargs' (or null), and 'dependencies' (array of string step ids it depends on).\n"
            "- 'acceptance': an array of string acceptance criteria\n"
            "Only output the JSON object, nothing else."
        )

        if mental_state:
            bias_instructions = []
            if mental_state.optimism > 0.7:
                bias_instructions.append("You are feeling very optimistic. You can take more risks and propose more ambitious steps.")
            elif mental_state.optimism < 0.3:
                bias_instructions.append("You are feeling pessimistic. Be cautious and prioritize safety and verification in your plan.")

            if mental_state.fear > 0.6:
                bias_instructions.append("FEAR LEVEL HIGH: You are feeling anxious about potential failures. You MUST include at least one explicit verification step for every state-changing action.")

            if mental_state.overconfidence > 0.7:
                bias_instructions.append("You are very confident in your abilities. You might skip some redundant verification steps or combine multiple complex operations.")

            if bias_instructions:
                system_prompt += "\n\nCognitive Bias Modifiers:\n" + "\n".join(bias_instructions)

        if behavior_weights:
            weight_instructions = ["\n\nBehavioral Parameters:"]
            if behavior_weights.get("exploration", 1.0) > 1.3:
                weight_instructions.append("- High exploration mode: Prefer new or complex tools, suggest experimental steps.")
            elif behavior_weights.get("exploration", 1.0) < 0.7:
                weight_instructions.append("- Low exploration mode: Stick to proven, safe tools and standard procedures.")

            if behavior_weights.get("directness", 1.0) > 1.3:
                weight_instructions.append("- High directness: Prioritize efficient, minimal-step plans.")
            elif behavior_weights.get("directness", 1.0) < 0.7:
                weight_instructions.append("- Low directness: Include extra verification and safety checkpoints in the plan.")

            if len(weight_instructions) > 1:
                system_prompt += "\n".join(weight_instructions)

        if skill_weights:
            top_skills = sorted(skill_weights.items(), key=lambda x: x[1], reverse=True)[:3]
            preferred = [s for s, w in top_skills if w > 1.2]
            if preferred:
                system_prompt += f"\n\nHighly weighted skills for this user: {', '.join(preferred)}. Prefer using these if appropriate."

        if self.habit_tracker:
            suggested_strategy = self.habit_tracker.suggest_strategy(user_input, user_id=user_id)
            if suggested_strategy:
                system_prompt += f"\n\nSuggested strategy based on past success: consider using the '{suggested_strategy}' skill."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        try:
            response_text = await self.llm.chat_completion(messages)
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            plan_dict = json.loads(response_text.strip())
            try:
                typed_plan = TypedPlan(**plan_dict)
            except ValidationError as ve:
                logging.error(f"Plan validation failed: {ve}")
                self.clear_pending_plan(user_id=user_id)
                return []

            plan_steps = [step.model_dump() for step in typed_plan.steps]
            for i, step in enumerate(plan_steps):
                if not step.get("id"):
                    step["id"] = f"step_{i}"

            for i, step in enumerate(plan_steps):
                if step["skill"] is not None and not self.skills.has_skill(step["skill"]):
                    logging.error(f"Step {i} uses unknown skill: {step['skill']}.")
                    self.clear_pending_plan(user_id=user_id)
                    return []
                if step["skill_kwargs"] is not None and not isinstance(step["skill_kwargs"], dict):
                    logging.error(f"Step {i} 'skill_kwargs' must be a dictionary or null.")
                    self.clear_pending_plan(user_id=user_id)
                    return []

            try:
                plan_steps = DependencyGraph.topological_sort(plan_steps)
            except ValueError as ve:
                logging.error(f"Plan cycle validation failed: {ve}")
                self.clear_pending_plan(user_id=user_id)
                return []

            state.current_plan = plan_steps
            state.completed_steps = []
            state.current_goal = typed_plan.goal
            state.current_constraints = typed_plan.constraints
            state.current_risk = typed_plan.risk
            state.current_acceptance = typed_plan.acceptance
            return plan_steps
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode plan JSON: {e}")
            self.clear_pending_plan(user_id=user_id)
            return []
        except Exception as e:
            logging.error(f"Error during plan generation: {e}")
            self.clear_pending_plan(user_id=user_id)
            return []

    async def spawn_sub_agent(self, task: str, context: str) -> str:
        logging.info(f"Planner spawning sub-agent for task: {task[:50]}")
        manager = TeamManager(llm=self.llm)
        results = await manager.spawn_and_execute([{'description': task}], context=context)
        return results[0] if results else 'Error: No result from subagent.'
        logging.info(f"Planner spawning sub-agent for task: {task[:50]}")

    def get_current_plan(self, user_id: Optional[Any] = None) -> List[Dict[str, Any]]:
        return self.get_user_state(user_id).current_plan

    def get_completed_steps(self, user_id: Optional[Any] = None) -> List[Dict[str, Any]]:
        return self.get_user_state(user_id).completed_steps

    def mark_step_completed(self, step_index: int, result: str, user_id: Optional[Any] = None) -> None:
        state = self.get_user_state(user_id)
        if 0 <= step_index < len(state.current_plan):
            step = state.current_plan.pop(step_index)
            step['result'] = result
            state.completed_steps.append(step)
            logging.info(f"Step completed: {step.get('description')}")
        else:
            logging.warning(f"Invalid step index: {step_index}")

    def mark_step_id_completed(self, step_id: str, result: str, user_id: Optional[Any] = None) -> None:
        state = self.get_user_state(user_id)
        for i, step in enumerate(state.current_plan):
            if step.get("id") == step_id:
                step = state.current_plan.pop(i)
                step['result'] = result
                state.completed_steps.append(step)
                logging.info(f"Step completed by ID: {step_id} - {step.get('description')}")
                return
        logging.warning(f"Step ID not found in current plan: {step_id}")

    def get_executable_steps(self, user_id: Optional[Any] = None) -> List[Dict[str, Any]]:
        state = self.get_user_state(user_id)
        completed_ids: Set[str] = {step.get("id") for step in state.completed_steps if step.get("id")}
        return DependencyGraph.get_executable_steps(state.current_plan, completed_ids)

    def get_state_summary(self, user_id: Optional[Any] = None) -> str:
        state = self.get_user_state(user_id)
        summary = "Planner State:\n"
        if not state.current_plan and not state.completed_steps:
            return summary + "  No active plan."

        if state.current_goal:
            summary += f"  Goal: {state.current_goal}\n"
            summary += f"  Risk: {state.current_risk}\n"
            if state.current_constraints:
                summary += f"  Constraints: {', '.join(state.current_constraints)}\n"

        if state.completed_steps:
            summary += "  Completed Steps:\n"
            for step in state.completed_steps:
                summary += f"    - {step.get('description')} (Skill: {step.get('skill')}) -> {step.get('result')}\n"

        if state.current_plan:
            summary += "  Pending Steps:\n"
            for step in state.current_plan:
                deps = f" [deps: {', '.join(step.get('dependencies', []))}]" if step.get('dependencies') else ""
                summary += f"    - {step.get('id')}: {step.get('description')} (Skill: {step.get('skill')}){deps}\n"
        return summary

    def pause_current_plan(self, user_id: Optional[Any] = None) -> None:
        state = self.get_user_state(user_id)
        if not state.current_plan and not state.completed_steps:
            logging.warning("No active plan to pause.")
            return

        state.paused_plan = {
            "current_plan": state.current_plan,
            "completed_steps": state.completed_steps,
            "current_goal": state.current_goal,
            "current_constraints": state.current_constraints,
            "current_risk": state.current_risk,
            "current_acceptance": state.current_acceptance
        }
        state.current_plan = []
        state.completed_steps = []
        state.current_goal = None
        state.current_constraints = []
        state.current_risk = None
        state.current_acceptance = []
        logging.info("Current plan paused.")

    def resume_plan(self, user_id: Optional[Any] = None) -> bool:
        state = self.get_user_state(user_id)
        if not state.paused_plan:
            logging.warning("No paused plan to resume.")
            return False

        state.current_plan = state.paused_plan.get("current_plan", [])
        state.completed_steps = state.paused_plan.get("completed_steps", [])
        state.current_goal = state.paused_plan.get("current_goal")
        state.current_constraints = state.paused_plan.get("current_constraints", [])
        state.current_risk = state.paused_plan.get("current_risk")
        state.current_acceptance = state.paused_plan.get("current_acceptance", [])
        state.paused_plan = None
        logging.info("Paused plan resumed.")
        return True

    def clear_pending_plan(self, user_id: Optional[Any] = None) -> None:
        state = self.get_user_state(user_id)
        state.current_plan = []
        state.current_goal = None
        state.current_constraints = []
        state.current_risk = None
        state.current_acceptance = []
        state.paused_plan = None
        logging.info("Pending plan steps cleared.")
