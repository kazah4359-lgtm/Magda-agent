import logging
from typing import List, Dict, Optional, Any

from magda_agent.llm_client import LLMClient
from magda_agent.memory.procedural import ProceduralMemory

class SkillExtractor:
    """
    Extracts successful procedures into new skills from past executions.
    """
    def __init__(self, procedural_memory: ProceduralMemory, llm: LLMClient) -> None:
        self.procedural_memory = procedural_memory
        self.llm = llm

    async def extract_skill(
        self,
        task_description: str,
        experience_logs: List[Dict[str, Any]],
        user_id: Optional[int] = None
    ) -> None:
        """
        Extracts a reusable skill from experience logs.
        """
        logs_text = ""
        for i, log in enumerate(experience_logs):
            action = log.get("action", "No action")
            outcome = log.get("outcome", "No outcome")
            logs_text += f"Step {i+1}: {action} -> {outcome}\n"

        prompt = f"""
        Extract a reusable procedural skill based on the following successful task execution experience.

        Task: {task_description}

        Experience Logs:
        {logs_text}

        Provide ONLY the reusable procedure text.
        """

        try:
            response = await self.llm.chat_completion([{"role": "user", "content": prompt}])
            procedure_text = response.strip()

            if procedure_text:
                self.procedural_memory.store_procedure(
                    name="extracted_skill",
                    procedure=procedure_text,
                    user_id=user_id,
                    metadata={"source_task": task_description}
                )
                logging.info(f"Extracted and stored new skill from task: {task_description[:30]}...")
            else:
                logging.warning("LLM generated an empty skill extraction.")
        except Exception as e:
            logging.error(f"Failed to extract skill: {e}")
