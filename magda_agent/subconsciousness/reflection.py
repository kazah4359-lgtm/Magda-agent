import asyncio
import logging
from typing import List
from magda_agent.llm_client import LLMClient
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.memory.storage import MemorySystem

class Subconsciousness:
    """
    Background processes for self-reflection and memory consolidation.
    Simulates the "subconscious" mind working while not directly interacting.
    """
    def __init__(
        self,
        llm: LLMClient,
        emotions: EmotionalEngine,
        memory: MemorySystem,
        interval: int = 60  # Reflection interval in seconds
    ):
        self.llm = llm
        self.emotions = emotions
        self.memory = memory
        self.interval = interval
        self.is_running = False

    async def start(self):
        """Start the background reflection loop."""
        self.is_running = True
        logging.info("Subconsciousness reflection loop started.")
        while self.is_running:
            await asyncio.sleep(self.interval)
            await self.reflect()

    async def stop(self):
        self.is_running = False
        logging.info("Subconsciousness reflection loop stopped.")

    async def reflect(self):
        """
        Perform a cycle of self-reflection.
        Analyzes recent memories, adjusts emotions, and consolidates memory.
        """
        logging.info("Subconsciousness is reflecting...")

        recent_memories = self.memory.short_term
        if not recent_memories:
            return

        # 1. Consolidate memory (standard logic)
        self.memory.consolidate()

        # 2. Self-Reflection reasoning
        # Magda looks at her own performance and feels "proud" or "worried"
        memories_text = "\n".join([m.content for m in recent_memories[-3:]])

        prompt = f"""
        Recent events:
        {memories_text}

        Based on these events, perform a brief self-reflection.
        How are you doing? Are you fulfilling your goals as an AGI?
        Suggest a minor adjustment to your emotional state (Pleasure, Arousal, Dominance) as a result of this reflection.
        Return your reflection and the PAD adjustment.
        """

        # We don't want to spam the LLM too much, but for PoC we do it once per reflection cycle
        reflection = await self.llm.chat_completion([{"role": "system", "content": "You are Magda's subconscious mind."}, {"role": "user", "content": prompt}])

        logging.info(f"Reflection result: {reflection}")

        # 3. Apply emotional "reward" or "punishment" based on reflection
        # For PoC, we just slightly increase dominance to simulate "self-growth"
        self.emotions.update(0.02, -0.01, 0.05)

        self.memory.add_memory(
            content=f"Subconscious reflection: {reflection}",
            importance=0.4,
            emotional_state=self.emotions.state,
            tags=["reflection", "internal"]
        )
