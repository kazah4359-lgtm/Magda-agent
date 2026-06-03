import logging
from typing import List, Dict, Any, Optional
from magda_agent.llm_client import LLMClient
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.memory.storage import MemorySystem
from magda_agent.skills.registry import SkillRegistry

class Consciousness:
    """
    The main cognitive loop of the AGI agent.
    Responsible for perception, memory retrieval, emotional processing, and response generation.
    """
    def __init__(
        self,
        llm: LLMClient,
        emotions: EmotionalEngine,
        memory: MemorySystem,
        skills: SkillRegistry
    ):
        self.llm = llm
        self.emotions = emotions
        self.memory = memory
        self.skills = skills

    async def process_input(self, user_input: str) -> str:
        logging.info(f"Consciousness processing: {user_input}")

        # 1. Perception & Emotion Update (Initial reaction)
        # For simplicity, we just slightly increase arousal when receiving input
        self.emotions.update(0.01, 0.05, 0.01)

        # 2. Memory Retrieval
        relevant_memories = self.memory.retrieve_relevant(user_input)
        context_str = "\n".join([f"- {m.content}" for m in relevant_memories])

        # 3. LLM Reasoning
        system_prompt = self.llm.get_system_prompt(
            context=context_str,
            emotions=self.emotions.get_summary()
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        # Determine if a skill should be used (Simplified logic for PoC)
        # In a real system, the LLM would decide which tool to call.
        response = await self.llm.chat_completion(messages)

        # 4. Post-processing & Memory Storage
        self.memory.add_memory(
            content=f"User said: {user_input} | I replied: {response}",
            importance=0.5,
            emotional_state=self.emotions.state,
            tags=["conversation"]
        )

        # Gradual emotional decay after processing
        self.emotions.decay()

        return response

    def get_internal_state(self) -> str:
        return f"""
{self.emotions.get_summary()}
{self.memory.get_summary()}
{self.skills.get_skills_summary()}
"""
