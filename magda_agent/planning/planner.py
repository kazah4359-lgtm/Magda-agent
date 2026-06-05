import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from magda_agent.llm_client import LLMClient
from magda_agent.skills.registry import SkillRegistry
from magda_agent.learning.habits import HabitTracker

class PlanStep(BaseModel):
    description: str
    skill: Optional[str] = None
    skill_kwargs: Optional[Dict[str, Any]] = None

class TypedPlan(BaseModel):
    goal: str
    constraints: List[str] = Field(default_factory=list)
    risk: str
    steps: List[PlanStep]
    acceptance: List[str] = Field(default_factory=list)

class Planner:
    """
    Prefrontal Cortex (Planner) module.
    Responsible for breaking down complex queries into steps (plans),
    selecting which skills to use, and maintaining state.
    """

    def __init__(self, llm: LLMClient, skills: SkillRegistry, habit_tracker: Optional[HabitTracker] = None):
        self.llm = llm
        self.skills = skills
        self.habit_tracker = habit_tracker
        self.current_plan: List[Dict[str, Any]] = []
        self.completed_steps: List[Dict[str, Any]] = []
        self.current_goal: Optional[str] = None
        self.current_constraints: List[str] = []
        self.current_risk: Optional[str] = None
        self.current_acceptance: List[str] = []

    async def generate_plan(self, user_input: str, user_id: int = None) -> List[Dict[str, Any]]:
        """
        Analyzes the user input and generates a sequence of steps.
        Each step may use a skill.

        Args:
            user_input (str): The prompt from the user.
            user_id (int, optional): The ID of the user.

        Returns:
            List[Dict[str, Any]]: A list of steps forming the plan.
        """
        logging.info("Generating plan for input")

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
            "- 'steps': an array of step objects. Each step must have 'description', 'skill' (or null), and 'skill_kwargs' (or null).\n"
            "- 'acceptance': an array of string acceptance criteria\n"
            "Only output the JSON object, nothing else."
        )

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

            # Basic cleanup in case the LLM returned markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            plan_dict = json.loads(response_text)
            try:
                typed_plan = TypedPlan(**plan_dict)
            except ValidationError as ve:
                logging.error(f"Plan validation failed: {ve}")
                self.clear_pending_plan()
                return []

            plan_steps = [step.model_dump() for step in typed_plan.steps]

            for i, step in enumerate(plan_steps):
                if step["skill"] is not None and not self.skills.has_skill(step["skill"]):
                    logging.error(f"Step {i} uses unknown skill: {step['skill']}.")
                    self.clear_pending_plan()
                    return []
                if step["skill_kwargs"] is not None and not isinstance(step["skill_kwargs"], dict):
                    logging.error(f"Step {i} 'skill_kwargs' must be a dictionary or null.")
                    self.clear_pending_plan()
                    return []

            self.current_plan = plan_steps
            self.completed_steps = []
            self.current_goal = typed_plan.goal
            self.current_constraints = typed_plan.constraints
            self.current_risk = typed_plan.risk
            self.current_acceptance = typed_plan.acceptance
            return plan_steps
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode plan JSON: {e}")
            self.clear_pending_plan()
            return []
        except Exception as e:
            logging.error(f"Error during plan generation: {e}")
            self.clear_pending_plan()
            return []

    def get_current_plan(self) -> List[Dict[str, Any]]:
        """
        Returns the currently active plan.

        Returns:
            List[Dict[str, Any]]: The current sequence of pending steps.
        """
        return self.current_plan

    def mark_step_completed(self, step_index: int, result: str) -> None:
        """
        Marks a specific step as completed, storing its result.

        Args:
            step_index (int): The index of the step in the current plan.
            result (str): The outcome or result of executing the step.
        """
        if 0 <= step_index < len(self.current_plan):
            step = self.current_plan.pop(step_index)
            step['result'] = result
            self.completed_steps.append(step)
            logging.info(f"Step completed: {step.get('description')}")
        else:
            logging.warning(f"Invalid step index: {step_index}")

    def get_state_summary(self) -> str:
        """
        Returns a summary of the current planner state.

        Returns:
            str: A formatted string describing pending and completed steps.
        """
        summary = "Planner State:\n"
        if not self.current_plan and not self.completed_steps:
            return summary + "  No active plan."

        if self.current_goal:
            summary += f"  Goal: {self.current_goal}\n"
            summary += f"  Risk: {self.current_risk}\n"
            if self.current_constraints:
                summary += f"  Constraints: {', '.join(self.current_constraints)}\n"

        if self.completed_steps:
            summary += "  Completed Steps:\n"
            for step in self.completed_steps:
                summary += f"    - {step.get('description')} (Skill: {step.get('skill')}) -> {step.get('result')}\n"

        if self.current_plan:
            summary += "  Pending Steps:\n"
            for step in self.current_plan:
                summary += f"    - {step.get('description')} (Skill: {step.get('skill')})\n"

        return summary

    def clear_pending_plan(self) -> None:
        """
        Clears the current pending plan steps.
        """
        self.current_plan = []
        self.current_goal = None
        self.current_constraints = []
        self.current_risk = None
        self.current_acceptance = []
        logging.info("Pending plan steps cleared.")
