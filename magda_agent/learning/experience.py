import logging
from typing import List, Dict, Optional, Any
import re

from magda_agent.llm_client import LLMClient
from magda_agent.memory.procedural import ProceduralMemory

class ExperienceToSkillCreator:
    """
    Creates reusable skills from experience, inspired by Hermes.
    """
    def __init__(self, procedural_memory: ProceduralMemory, llm: LLMClient) -> None:
        self.procedural_memory = procedural_memory
        self.llm = llm

    async def create_skill_from_experience(
        self,
        task_description: str,
        execution_trace: List[Dict[str, Any]],
        user_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Analyzes an execution trace and generates a Python skill.
        Returns the skill code if successful, else None.
        """
        trace_text = ""
        for i, step in enumerate(execution_trace):
            action = step.get("action", "unknown action")
            outcome = step.get("outcome", "unknown outcome")
            trace_text += f"Step {i+1}: {action} -> {outcome}\n"

        prompt = f"""
        Analyze the following execution trace of a successful task.
        If the steps represent a highly reusable procedure, generate a Python function (a 'skill') that encapsulates this logic.
        Return ONLY valid Python code starting with 'def '. Do not include markdown formatting or explanations.
        If it's not suitable for a skill, return exactly 'NOT_REUSABLE'.

        Task: {task_description}

        Execution Trace:
        {trace_text}
        """

        try:
            response = await self.llm.chat_completion([{"role": "user", "content": prompt}])
            code = response.strip()

            if code and code != "NOT_REUSABLE" and code.startswith("def "):
                match = re.search(r"def\s+([a-zA-Z0-9_]+)\(", code)
                skill_name = match.group(1) if match else "generated_skill"

                self.procedural_memory.store_procedure(
                    name=skill_name,
                    procedure=code,
                    user_id=user_id,
                    metadata={"source_task": task_description, "type": "python_skill_experience"}
                )
                logging.info(f"Dynamically generated and stored new Python skill from experience: {skill_name}")
                return code
            else:
                logging.info("Experience not suitable for skill generation.")
                return None
        except Exception as e:
            logging.error(f"Failed to create skill from experience: {e}")
            return None
