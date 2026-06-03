import asyncio
import os
import json
from openai import AsyncOpenAI
from .emotions import Personality
from magda_agent.memory.memory_manager import MemoryManager

class Subconscious:
    """
    Фоновые процессы: консолидация памяти, рефлексия, расчет эмоций.
    Оно наблюдает за Сознанием и корректирует драйвы.
    """
    def __init__(self, personality: Personality, memory: MemoryManager):
        self.personality = personality
        self.memory = memory
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def run_background_reflection(self):
        """Бесконечный фоновый цикл подсознания (Сон / Рефлексия)."""
        while True:
            await asyncio.sleep(300) # Раз в 5 минут

            # Если есть короткая память, подсознание "переваривает" её
            recent_context = self.memory.get_short_term_context()
            if recent_context:
                # Интеллектуальный анализ диалога через LLM
                await self._reflect_on_context(recent_context)
            else:
                # Если памяти нет, растет страх одиночества
                self.personality.update_drive("fear", 0.5)

    async def _reflect_on_context(self, context: list):
        """Анализирует контекст диалога с помощью LLM и меняет эмоции."""
        dialogue = "\n".join([f"{m['role']}: {m['content']}" for m in context])
        prompt = (
            "You are the subconscious of the AI Magda. Analyze the following recent dialogue. "
            "How should Magda's emotional drives change based on this interaction?\n"
            "Drives to adjust: 'love' (affection), 'passion' (motivation), 'fear' (anxiety/errors), 'boredom', 'loneliness'.\n"
            "Respond ONLY with a valid JSON object where keys are drive names and values are floats representing the change delta (-10.0 to 10.0).\n\n"
            f"Dialogue:\n{dialogue}"
        )

        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            changes = json.loads(response.choices[0].message.content)
            for drive, delta in changes.items():
                if isinstance(delta, (int, float)):
                    self.personality.update_drive(drive, float(delta))
            print(f"Subconscious reflection applied changes: {changes}")
        except Exception as e:
            print(f"Subconscious reflection error: {e}")
