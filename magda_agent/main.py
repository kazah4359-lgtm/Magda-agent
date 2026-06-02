import asyncio
import os
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from dotenv import load_dotenv

from cognitive_core.consciousness import Consciousness

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TEST_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher()

# Инициализация когнитивного ядра
magda_consciousness = Consciousness()

@dp.message(CommandStart())
async def send_welcome(message: types.Message):
    await message.answer("Привет! Я Magda, когнитивный ИИ-агент. Я проснулась.")

@dp.message()
async def handle_message(message: types.Message):
    # Передаем сообщение в Сознание агента
    response = magda_consciousness.process_input(message.text)
    await message.answer(response)

async def main():
    logging.basicConfig(level=logging.INFO)
    print("Starting Magda Agent Dashboard (Telegram Bot)...")
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
