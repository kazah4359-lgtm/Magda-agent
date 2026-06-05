import logging
from typing import List, Optional

from magda_agent.llm_client import LLMClient
from magda_agent.memory.procedural import ProceduralMemory


class TaskRecoveryLessons:
    """
    Mechanism to convert PR failures and execution errors into reusable 'lessons'
    in Procedural Memory, minimizing repeated identical mistakes.
    """

    def __init__(self, procedural_memory: ProceduralMemory, llm: LLMClient) -> None:
        """
        Initializes the TaskRecoveryLessons module.

        Args:
            procedural_memory (ProceduralMemory): The procedural memory storage.
            llm (LLMClient): The LLM client to summarize failures into lessons.
        """
        self.procedural_memory = procedural_memory
        self.llm = llm

    async def generate_and_store_lesson(
        self,
        task_description: str,
        failure_reason: str,
        user_id: Optional[int] = None
    ) -> None:
        """
        Summarizes the task and failure into a concise lesson using the LLM,
        and stores it in Procedural Memory.

        Args:
            task_description (str): A description of the task that failed.
            failure_reason (str): The details or logs of the failure.
            user_id (int, optional): The ID of the user.
        """
        prompt = f"""
        Analyze the following task failure and generate a concise, reusable lesson or anti-pattern.
        The lesson should describe what went wrong and what should be done differently next time.
        Keep it under 3 sentences.

        Task Description:
        {task_description}

        Failure Reason:
        {failure_reason}

        Lesson:
        """

        try:
            response = await self.llm.chat_completion(
                [{"role": "user", "content": prompt}]
            )
            lesson_text = response.strip()

            if lesson_text:
                self.procedural_memory.store_procedure(
                    "recovery_lesson",
                    lesson_text,
                    user_id=user_id
                )
                logging.info(f"Stored recovery lesson for task: {task_description[:30]}...")
            else:
                logging.warning("LLM generated an empty lesson.")
        except Exception as e:
            logging.error(f"Failed to generate and store lesson: {e}")

    def retrieve_relevant_lessons(
        self,
        task_description: str,
        user_id: Optional[int] = None
    ) -> List[str]:
        """
        Recalls relevant past lessons based on a new task description.

        Args:
            task_description (str): The description of the new task.
            user_id (int, optional): The ID of the user.

        Returns:
            List[str]: A list of relevant procedural lessons.
        """
        return self.procedural_memory.recall_procedure(task_description, user_id=user_id)
