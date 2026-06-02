import asyncio
import os
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from dotenv import load_dotenv

from cognitive_core.consciousness import Consciousness
from cognitive_core.subconscious import Subconscious

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TEST_TOKEN")
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID", None)

bot = Bot(token=TOKEN)
dp = Dispatcher()

# Инициализация когнитивного ядра
magda_consciousness = Consciousness()
magda_subconscious = Subconscious(magda_consciousness.personality, magda_consciousness.memory)

# Сохраняем последний chat_id для проактивных сообщений
# (В реальной системе это должно храниться в БД)
current_chat_id = None

@dp.message(CommandStart())
async def send_welcome(message: types.Message):
    if ALLOWED_USER_ID and str(message.from_user.id) != str(ALLOWED_USER_ID):
        await message.answer("Извините, я общаюсь только со своим создателем.")
        return

    global current_chat_id
    current_chat_id = message.chat.id
    await message.answer("Привет! Я Magda, когнитивный ИИ-агент. Я проснулась.")

@dp.message()
async def handle_message(message: types.Message):
    if ALLOWED_USER_ID and str(message.from_user.id) != str(ALLOWED_USER_ID):
        return

    global current_chat_id
    current_chat_id = message.chat.id

    # Сброс скуки при общении
    magda_consciousness.personality.update_drive("boredom", -20.0)
    magda_consciousness.personality.update_drive("loneliness", -20.0)

    # Асинхронно передаем сообщение в Сознание агента
    response = await magda_consciousness.process_input_async(message.text)
    await message.answer(response)

async def proactive_pulse():
    """
    Пульс (Heartbeat) - фоновый процесс, имитирующий течение времени.
    Каждую минуту агент немного скучает и чувствует одиночество, если с ним не говорят.
    """
    while True:
        await asyncio.sleep(60) # раз в минуту

        # Растет скука и одиночество
        magda_consciousness.personality.update_drive("boredom", 2.0)
        magda_consciousness.personality.update_drive("loneliness", 1.0)

        # Если скука превысила порог, агент может проявить инициативу
        if magda_consciousness.personality.drives["boredom"] > 80.0 and current_chat_id:
            msg = await magda_consciousness.generate_proactive_message()
            if msg:
                await bot.send_message(current_chat_id, msg)
                # Сбрасываем скуку после проявления инициативы
                magda_consciousness.personality.update_drive("boredom", -40.0)


async def main():
    logging.basicConfig(level=logging.INFO)
    print("Starting Magda Agent Dashboard (Telegram Bot)...")

    # Запускаем фоновый пульс
    asyncio.create_task(proactive_pulse())
    # Запускаем подсознание
    asyncio.create_task(magda_subconscious.run_background_reflection())

    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
