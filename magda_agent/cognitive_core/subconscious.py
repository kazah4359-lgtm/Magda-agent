import asyncio
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

    async def run_background_reflection(self):
        """Бесконечный фоновый цикл подсознания (Сон / Рефлексия)."""
        while True:
            await asyncio.sleep(300) # Раз в 5 минут

            # Если есть короткая память, подсознание "переваривает" её
            recent_context = self.memory.get_short_term_context()
            if recent_context:
                # В будущем здесь можно добавить запрос к LLM для оценки качества диалога
                # Пока что просто поощряем за общение
                self.personality.update_drive("love", 1.0)
            else:
                # Если памяти нет, растет страх одиночества
                self.personality.update_drive("fear", 0.5)
