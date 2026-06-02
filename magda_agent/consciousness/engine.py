import asyncio
import os
from typing import List, Dict
from openai import AsyncOpenAI
from magda_agent.memory.long_term_memory import LongTermMemory
from magda_agent.emotions.emotional_engine import EmotionalEngine

class Consciousness:
    def __init__(self, memory: LongTermMemory, emotions: EmotionalEngine):
        self.memory = memory
        self.emotions = emotions
        self.current_focus: str = "Idle"
        self.working_memory: List[str] = []
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy_key"))

    async def process_input(self, user_input: str) -> str:
        # 1. Update focus
        self.current_focus = f"Processing: {user_input[:20]}..."

        # 2. Retrieve relevant memories
        tags = user_input.lower().split()
        relevant_memories = self.memory.search_memories(tags)

        # 3. Formulate context
        context_str = "\n".join([m.content for m in relevant_memories[:5]])
        mood = self.emotions.get_mood_label()

        system_prompt = (
            f"You are Magdalina, an AGI agent with a cognitive architecture.\n"
            f"Current Mood: {mood}\n"
            f"Recent relevant memories:\n{context_str}\n"
            f"Respond to the user naturally, taking your mood and memories into account."
        )

        # 4. "Think" using OpenAI if key is provided
        if os.getenv("OPENAI_API_KEY"):
            try:
                chat_completion = await self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    model="gpt-4o-mini",
                )
                response = chat_completion.choices[0].message.content
            except Exception as e:
                response = f"Error during thought process: {e}"
        else:
            response = f"[Dry Run Mode] [Mood: {mood}] I don't have an API key, but I'm thinking about: {user_input}"

        # 5. Store the interaction in memory
        self.memory.add_memory(
            content=f"User said: {user_input}. I responded: {response}",
            importance=0.5,
            emotion=self.emotions.get_state(),
            tags=tags
        )

        # 6. Affect emotions (Simple example: interaction increases arousal)
        self.emotions.update({"arousal": 0.1, "pleasure": 0.05})

        return response

    def get_status(self) -> Dict:
        return {
            "focus": self.current_focus,
            "mood": self.emotions.get_mood_label(),
            "emotional_state": self.emotions.get_state(),
            "working_memory_size": len(self.working_memory)
        }
