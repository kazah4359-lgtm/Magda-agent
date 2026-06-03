import json
import logging
from typing import List, Dict, Any, Optional
from magda_agent.llm_client import LLMClient
from magda_agent.skills.registry import SkillRegistry

class Planner:
    """
    Prefrontal Cortex (Planner) module.
    Responsible for breaking down complex queries into steps (plans),
    selecting which skills to use, and maintaining state.
    """

    def __init__(self, llm: LLMClient, skills: SkillRegistry):
        self.llm = llm
        self.skills = skills
        self.current_plan: List[Dict[str, Any]] = []
        self.completed_steps: List[Dict[str, Any]] = []

    async def generate_plan(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Analyzes the user input and generates a sequence of steps.
        Each step may use a skill.

        Args:
            user_input (str): The prompt from the user.

        Returns:
            List[Dict[str, Any]]: A list of steps forming the plan.
        """
        logging.info("Generating plan for input")

        skills_desc = self.skills.get_skills_summary()

        system_prompt = (
            "You are the Prefrontal Cortex of an AI agent. Your job is to break down "
            "the user's request into a logical sequence of steps. "
            "Available skills:\n"
            f"{skills_desc}\n"
            "Return a JSON array of steps. Each step must be a JSON object with keys: "
            "'description' (what to do) and 'skill' (the name of the skill to use, or null if none). "
            "Only output the JSON array, nothing else."
        )

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

            plan = json.loads(response_text)
            if isinstance(plan, list):
                self.current_plan = plan
                self.completed_steps = []
                return plan
            else:
                logging.error("Plan generated is not a JSON list.")
                self.current_plan = []
                return []
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode plan JSON: {e}")
            self.current_plan = []
            return []
        except Exception as e:
            logging.error(f"Error during plan generation: {e}")
            self.current_plan = []
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

        if self.completed_steps:
            summary += "  Completed Steps:\n"
            for step in self.completed_steps:
                summary += f"    - {step.get('description')} (Skill: {step.get('skill')}) -> {step.get('result')}\n"

        if self.current_plan:
            summary += "  Pending Steps:\n"
            for step in self.current_plan:
                summary += f"    - {step.get('description')} (Skill: {step.get('skill')})\n"

        return summary
