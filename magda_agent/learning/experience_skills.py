import logging
import re
from typing import List, Dict, Optional, Any

from magda_agent.llm_client import LLMClient
from magda_agent.memory.procedural import ProceduralMemory


class HermesSkillExperienceManager:
    """
    Hermes-style Skill Creation from Experience.

    This module analyzes execution trace history to automatically synthesize
    and store new, highly reusable Python skills.
    """

    def __init__(self, llm: LLMClient, procedural_memory: ProceduralMemory) -> None:
        """
        Initializes the HermesSkillExperienceManager.

        Args:
            llm: The LLMClient instance to use for skill generation.
            procedural_memory: The ProceduralMemory instance to store generated skills.
        """
        self.llm = llm
        self.procedural_memory = procedural_memory

    async def create_skill_from_experience(
        self,
        task_description: str,
        execution_trace: List[Dict[str, Any]],
        user_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Parses an execution trace and synthesizes a new Python skill.

        Args:
            task_description: A description of the task that was completed.
            execution_trace: A list of dicts representing the steps taken during execution.
            user_id: The ID of the user for whom this skill should be stored.

        Returns:
            The generated Python code as a string if successful, or None otherwise.
        """
        if not execution_trace:
            return None

        trace_log = ""
        for i, step in enumerate(execution_trace):
            action = step.get("action", "unknown_action")
            outcome = step.get("outcome", "unknown_outcome")
            trace_log += f"Step {i+1}: {action} -> {outcome}\n"

        prompt = f"""
        Analyze the following execution trace of a successful task.
        If the steps represent a highly reusable procedure, generate a Python function (a 'skill') that encapsulates this logic.
        Return ONLY valid Python code starting with 'def '. Do not include markdown formatting or explanations.
        If it's not suitable for a skill, return exactly 'NOT_REUSABLE'.

        Task: {task_description}

        Execution Trace:
        {trace_log}
        """

        messages = [{"role": "user", "content": prompt}]
        try:
            response = await self.llm.chat_completion(messages)
            response = response.strip()

            # Remove potential markdown block wrappers
            if response.startswith("```python"):
                response = response[9:]
            elif response.startswith("```"):
                response = response[3:]

            if response.endswith("```"):
                response = response[:-3]

            response = response.strip()

            if response == "NOT_REUSABLE" or not response.startswith("def "):
                logging.info("Trace is not reusable as a skill.")
                return None

            # Extract function name
            match = re.search(r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", response)
            if not match:
                logging.warning("Could not extract function name from generated skill.")
                return None

            skill_name = match.group(1)

            metadata = {
                "source_task": task_description,
                "type": "hermes_experience_skill"
            }

            self.procedural_memory.store_procedure(
                name=skill_name,
                procedure=response,
                user_id=user_id,
                metadata=metadata
            )

            logging.info(f"Successfully generated and stored new skill: {skill_name}")
            return response

        except Exception as e:
            logging.error(f"Error during skill creation: {e}")
            return None
