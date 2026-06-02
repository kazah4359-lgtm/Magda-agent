import asyncio
import logging
from magda_agent.memory.long_term_memory import LongTermMemory
from magda_agent.emotions.emotional_engine import EmotionalEngine

class Subconsciousness:
    def __init__(self, memory: LongTermMemory, emotions: EmotionalEngine):
        self.memory = memory
        self.emotions = emotions
        self.is_running = False

    async def run_reflection_loop(self):
        """
        Background process that consolidates memory and decays emotions.
        """
        self.is_running = True
        logging.info("Subconsciousness reflection loop started.")
        while self.is_running:
            # 1. Consolidate memories (decay/cleanup)
            logging.debug("Subconsciousness: Consolidating memories...")
            self.memory.consolidate()

            # 2. Emotional decay
            logging.debug("Subconsciousness: Decaying emotions...")
            self.emotions.decay()

            # 3. "Dream" or reflect on lessons learned
            # In a more advanced version, this would use an LLM to summarize recent events

            # Sleep for a while (e.g., 1 hour in real life, but shorter for demo)
            await asyncio.sleep(60)

    def stop(self):
        self.is_running = False
