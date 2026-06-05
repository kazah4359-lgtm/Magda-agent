import logging
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
