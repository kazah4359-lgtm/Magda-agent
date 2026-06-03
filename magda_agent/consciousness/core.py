import logging
from typing import List, Dict, Any, Optional
from magda_agent.llm_client import LLMClient
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.memory.storage import MemorySystem
from magda_agent.skills.registry import SkillRegistry
from magda_agent.planning.planner import Planner
from magda_agent.memory.long_term import LongTermMemory

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
        skills: SkillRegistry,
        planner: Optional[Planner] = None,
        long_term_memory: Optional[LongTermMemory] = None
    ):
        self.llm = llm
        self.emotions = emotions
        self.memory = memory
        self.skills = skills
        self.planner = planner
        self.long_term_memory = long_term_memory

    async def process_input(self, user_input: str) -> str:
        logging.info(f"Consciousness processing: {user_input}")

        # 1. Perception & Emotion Update (Initial reaction)
        # For simplicity, we just slightly increase arousal when receiving input
        self.emotions.update(0.01, 0.05, 0.01)

        # 2. Memory Retrieval
        relevant_memories = self.memory.retrieve_relevant(user_input)
        context_str = "\n".join([f"- {m.content}" for m in relevant_memories])

        if self.long_term_memory:
            long_term_memories = self.long_term_memory.recall(user_input)
            if long_term_memories:
                context_str += "\nLong Term Memories:\n" + "\n".join([f"- {m}" for m in long_term_memories])

        # 3. Planning (Prefrontal Cortex)
        plan_str = ""
        if self.planner:
            # Only generate a new plan if we don't have an active one
            if not self.planner.get_current_plan():
                await self.planner.generate_plan(user_input)

            plan = self.planner.get_current_plan()
            if plan:
                plan_str = "Current Plan:\n" + "\n".join(
                    [f"- Step {i+1}: {step.get('description')} (Skill: {step.get('skill')})" for i, step in enumerate(plan)]
                )
                # In a real system we might execute the steps here and collect results.
                # For this task, integrating the planner before generation is required.

        # 4. LLM Reasoning
        system_prompt = self.llm.get_system_prompt(
            context=context_str,
            emotions=self.emotions.get_summary()
        )
        if plan_str:
            system_prompt += f"\n\n{plan_str}\nFollow the plan when generating the response."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        # Determine if a skill should be used (Simplified logic for PoC)
        # In a real system, the LLM would decide which tool to call.
        response = await self.llm.chat_completion(messages)

        # 5. Post-processing & Memory Storage
        memory_content = f"User said: {user_input} | I replied: {response}"
        self.memory.add_memory(
            content=memory_content,
            importance=0.5,
            emotional_state=self.emotions.state,
            tags=["conversation"]
        )

        if self.long_term_memory:
            self.long_term_memory.store(text=memory_content, metadata={"type": "conversation"})

        # Gradual emotional decay after processing
        self.emotions.decay()

        return response

    def get_internal_state(self) -> str:
        planner_state = self.planner.get_state_summary() if self.planner else "Planner: Not available"
        return f"""
{self.emotions.get_summary()}
{self.memory.get_summary()}
{self.skills.get_skills_summary()}
{planner_state}
"""
