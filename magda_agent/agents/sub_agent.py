import logging
from typing import Optional
from magda_agent.llm_client import LLMClient

class SubAgent:
    """
    SubAgent for executing isolated tasks in a separate context.
    Inspired by Claude Agent Teams and Hermes sub-agents.
    """
    def __init__(self, llm: LLMClient, system_prompt: Optional[str] = None):
        """
        Initializes the SubAgent.
        """
        self.llm = llm
        self.system_prompt = system_prompt or "You are an isolated Sub-Agent executing a specific task."

    async def execute(self, task: str, context: str) -> str:
        """
        Executes a task given the context.
        """
        logging.info(f"SubAgent starting task: {task[:50]}...")

        full_context = f"Parent Context:\n{context}\n\nAssigned Task:\n{task}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": full_context}
        ]

        try:
            result = await self.llm.chat_completion(messages)
            logging.info("SubAgent completed task.")
            return result
        except Exception as e:
            logging.error(f"Error executing SubAgent task: {e}")
            return f"Error executing SubAgent task: {e}"
