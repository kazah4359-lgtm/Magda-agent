import asyncio
import json
import logging
from typing import List
from magda_agent.llm_client import LLMClient
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.memory.storage import MemorySystem
from magda_agent.memory.procedural import ProceduralMemory
from magda_agent.memory.semantic import SemanticMemory

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
        semantic_memory: SemanticMemory = None,
        interval: int = 60  # Reflection interval in seconds
    ):
        self.llm = llm
        self.emotions = emotions
        self.memory = memory
        self.procedural_memory = procedural_memory
        self.semantic_memory = semantic_memory
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
            "new_facts": ["fact 1", "fact 2"],
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
        new_facts = []
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
            new_facts = parsed_data.get("new_facts", [])
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


        if self.semantic_memory:
            for fact in new_facts:
                existing = self.semantic_memory.search_facts(fact, top_k=3)
                if not existing:
                    self.semantic_memory.store_fact(fact)
                    continue

                # Check for conflicts
                existing_texts = "\n".join([f"ID: {e['id']} | Fact: {e['text']}" for e in existing])
                conflict_prompt = f"""
                New fact: '{fact}'
                Existing facts:
                {existing_texts}

                Does the new fact contradict any existing facts?
                Output ONLY a JSON object:
                {{
                    "conflict": true/false,
                    "conflicting_id": "ID of the conflicting fact or null",
                    "strategy": "newer_wins", // or "keep_old", "merge"
                    "resolved_fact": "The final fact to store if newer_wins or merge"
                }}
                """
                conflict_res = await self.llm.chat_completion([
                    {"role": "system", "content": "You detect semantic memory contradictions. Output ONLY JSON."},
                    {"role": "user", "content": conflict_prompt}
                ])
                try:
                    c_resp = conflict_res.strip()
                    if c_resp.startswith("```json"): c_resp = c_resp[7:]
                    if c_resp.startswith("```"): c_resp = c_resp[3:]
                    if c_resp.endswith("```"): c_resp = c_resp[:-3]
                    c_data = json.loads(c_resp.strip())

                    if c_data.get("conflict"):
                        logging.warning(f"Memory conflict detected! New: {fact}")
                        c_id = c_data.get("conflicting_id")
                        strategy = c_data.get("strategy")
                        resolved = c_data.get("resolved_fact")

                        if strategy in ["newer_wins", "merge"]:
                            if c_id:
                                self.semantic_memory.delete_fact(c_id)
                            self.semantic_memory.store_fact(resolved)
                    else:
                        self.semantic_memory.store_fact(fact)
                except Exception as e:
                    logging.error(f"Failed to process conflict resolution: {e}")
                    self.semantic_memory.store_fact(fact) # Fallback

        for task in proposed_tasks:
            logging.info(f"Subconscious proposed task: {task}")
