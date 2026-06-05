import asyncio
import json
import logging
from typing import List
from magda_agent.llm_client import LLMClient
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.memory.storage import MemorySystem
from magda_agent.memory.procedural import ProceduralMemory

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
        procedural_memory: ProceduralMemory = None,
        interval: int = 60  # Reflection interval in seconds
    ):
        self.llm = llm
        self.emotions = emotions
        self.memory = memory
        self.procedural_memory = procedural_memory
        self.interval = interval
        self.is_running = False
        self._stop_event = asyncio.Event()

    async def start(self):
        """Start the background reflection loop."""
        self.is_running = True
        self._stop_event.clear()
        logging.info("Subconsciousness reflection loop started.")
        while self.is_running:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                await self.reflect()

    async def stop(self):
        self.is_running = False
        self._stop_event.set()
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

        Based on these events, perform a structured self-reflection.
        How are you doing? Are you fulfilling your goals as an AGI?
        Identify any reusable lessons or anti-patterns. Are there tasks you should propose?
        Suggest a minor adjustment to your emotional state (Pleasure, Arousal, Dominance).

        You MUST respond ONLY with a valid JSON object in the exact format below, with no additional text:
        {{
            "summary": "Your textual self-reflection and summary here",
            "lessons": ["lesson 1", "lesson 2"],
            "anti_patterns": ["anti-pattern 1", "anti-pattern 2"],
            "proposed_tasks": ["proposed task 1", "proposed task 2"],
            "pad_adjustment": {{
                "p": 0.0,
                "a": 0.0,
                "d": 0.0
            }}
        }}
        """

        # We don't want to spam the LLM too much, but for PoC we do it once per reflection cycle
        raw_response = await self.llm.chat_completion([{"role": "system", "content": "You are Magda's subconscious mind. Always output valid JSON."}, {"role": "user", "content": prompt}])

        logging.info(f"Raw reflection response: {raw_response}")

        reflection_text = "Parsed reflection failed."
        lessons = []
        anti_patterns = []
        proposed_tasks = []
        p_adj, a_adj, d_adj = 0.02, -0.01, 0.05  # Defaults

        try:
            # Try to strip markdown if present
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            parsed_data = json.loads(cleaned_response.strip())
            reflection_text = parsed_data.get("summary", "No reflection text provided.")
            lessons = parsed_data.get("lessons", [])
            anti_patterns = parsed_data.get("anti_patterns", [])
            proposed_tasks = parsed_data.get("proposed_tasks", [])
            pad_adj = parsed_data.get("pad_adjustment", {})
            p_adj = float(pad_adj.get("p", 0.0))
            a_adj = float(pad_adj.get("a", 0.0))
            d_adj = float(pad_adj.get("d", 0.0))
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
            logging.error(f"Failed to parse subconscious reflection JSON: {e}")

        # 3. Apply emotional "reward" or "punishment" based on reflection
        self.emotions.update(p_adj, a_adj, d_adj)

        await self.memory.add_memory(
            content=f"Subconscious reflection: {reflection_text}",
            importance=0.4,
            emotional_state=self.emotions.state,
            tags=["reflection", "internal"]
        )

        if self.procedural_memory:
            for lesson in lessons:
                self.procedural_memory.store_procedure(name="lesson", procedure=lesson)
            for ap in anti_patterns:
                self.procedural_memory.store_procedure(name="anti_pattern", procedure=ap)

        for task in proposed_tasks:
            logging.info(f"Subconscious proposed task: {task}")
