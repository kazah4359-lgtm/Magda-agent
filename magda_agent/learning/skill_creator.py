import logging
import re
from typing import List, Dict, Optional, Any

from magda_agent.llm_client import LLMClient
from magda_agent.memory.procedural import ProceduralMemory

class SkillCreator:
    """
    Auto-creates reusable skills from successful multi-step task executions.
    Extracts the procedure using the LLM and stores it in ProceduralMemory.
    """
    def __init__(self, procedural_memory: ProceduralMemory, llm: LLMClient) -> None:
        self.procedural_memory = procedural_memory
        self.llm = llm

    async def extract_and_store_skill(
        self,
        task_description: str,
        execution_steps: List[Dict[str, Any]],
        user_id: Optional[int] = None
    ) -> None:
        """
        Analyzes a successful sequence of steps and extracts a reusable skill document.
        """
        steps_text = ""
        for i, step in enumerate(execution_steps):
            desc = step.get("description", "No description")
            skill = step.get("skill", "None")
            result = step.get("result", "No result")
            steps_text += f"Step {i+1}: {desc} (Used skill: {skill})\nResult: {result}\n\n"

        prompt = f"""
        Analyze the following successful multi-step task execution.
        Extract the core reusable procedure into a concise "skill candidate" document.
        Describe the steps taken and the overall strategy so that it can be reused in the future for similar tasks.

        Task: {task_description}

        Execution Trace:
        {steps_text}

        Provide ONLY the reusable procedure text. Keep it concise.
        """

        try:
            response = await self.llm.chat_completion([{"role": "user", "content": prompt}])
            procedure_text = response.strip()

            if procedure_text:
                self.procedural_memory.store_procedure(
                    name="skill_candidate",
                    procedure=procedure_text,
                    user_id=user_id,
                    metadata={"source_task": task_description}
                )
                logging.info(f"Created and stored new skill candidate from task: {task_description[:30]}...")
            else:
                logging.warning("LLM generated an empty skill procedure.")
        except Exception as e:
            logging.error(f"Failed to extract and store skill: {e}")

    async def analyze_chat_and_generate_skill(self, chat_history: List[Dict[str, str]], user_id: Optional[int] = None) -> None:
        """
        Analyzes chat history to detect repeated patterns and dynamically generates a Python skill.
        """
        chat_text = ""
        for i, msg in enumerate(chat_history):
            role = msg.get("role", "unknown")
            text = msg.get("content", "")
            chat_text += f"{role.capitalize()}: {text}\n"

        prompt = f"""
        Analyze the following chat history for repeated patterns in user requests.
        If you find a repeated pattern that can be automated, generate a Python function (a 'skill') that encapsulates this task.
        Return ONLY valid Python code starting with 'def '. Do not include markdown formatting or explanations.
        If no pattern is found, return exactly 'NO_PATTERN'.

        Chat History:
        {chat_text}
        """

        try:
            response = await self.llm.chat_completion([{"role": "user", "content": prompt}])
            code = response.strip()

            if code and code != "NO_PATTERN" and code.startswith("def "):
                # Extract function name from code
                match = re.search(r"def\s+([a-zA-Z0-9_]+)\(", code)
                skill_name = match.group(1) if match else "generated_skill"

                self.procedural_memory.store_procedure(
                    name=skill_name,
                    procedure=code,
                    user_id=user_id,
                    metadata={"source": "dynamic_generation", "type": "python_code"}
                )
                logging.info(f"Dynamically generated and stored new Python skill: {skill_name}")
            else:
                logging.info("No repeated pattern detected or failed to generate valid Python skill.")
        except Exception as e:
            logging.error(f"Failed to analyze chat and generate skill: {e}")
