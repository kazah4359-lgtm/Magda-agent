import logging
from typing import List, Dict, Optional, Any
import re

from magda_agent.llm_client import LLMClient
from magda_agent.memory.procedural import ProceduralMemory

class HermesSkillCreationLoopV2:
    """
    Agent creates new skills from past executions, inspired by Hermes.
    """
    def __init__(self, procedural_memory: ProceduralMemory, llm: LLMClient) -> None:
        self.procedural_memory = procedural_memory
        self.llm = llm

    async def analyze_and_create_skill(
        self,
        task_description: str,
        execution_trace: List[Dict[str, Any]],
        user_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Analyzes an execution trace and attempts to generate a Python skill.
        If a pattern is detected and valid Python code is generated, it is stored.
        Returns the name of the new skill if created, else None.
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
                # Extract function name from code
                match = re.search(r"def\s+([a-zA-Z0-9_]+)\(", code)
                skill_name = match.group(1) if match else "generated_hermes_skill"

                self.procedural_memory.store_procedure(
                    name=skill_name,
                    procedure=code,
                    user_id=user_id,
                    metadata={"source_task": task_description, "type": "python_skill_loop_v2"}
                )
                logging.info(f"Dynamically generated and stored new Python skill: {skill_name}")
                return skill_name
            else:
                logging.info("Execution trace not suitable for skill generation or generation failed.")
                return None
        except Exception as e:
            logging.error(f"Failed to analyze and create skill: {e}")
            return None
