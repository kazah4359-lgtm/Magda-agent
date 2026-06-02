import asyncio
import logging
import os
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

from magda_agent.memory.long_term_memory import LongTermMemory
from magda_agent.emotions.emotional_engine import EmotionalEngine
from magda_agent.consciousness.engine import Consciousness
from magda_agent.subconsciousness.reflexion import Subconsciousness
from magda_agent.skills.manager import SkillManager

# Initialize Bot and Dispatcher
BOT_TOKEN = os.getenv("BOT_TOKEN", "dummy_token")

# Core Modules
memory = LongTermMemory()
emotions = EmotionalEngine()
consciousness = Consciousness(memory, emotions)
subconsciousness = Subconsciousness(memory, emotions)
skills = SkillManager()

dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Hello, {message.from_user.full_name}! I am Magdalina, your AGI agent. I am now online with a full cognitive architecture.")

@dp.message(Command("status"))
async def command_status_handler(message: Message) -> None:
    status = consciousness.get_status()
    response = (
        f"<b>Magdalina Status:</b>\n"
        f"Focus: {status['focus']}\n"
        f"Mood: {status['mood']}\n"
        f"Emotional PAD: {status['emotional_state']}\n"
        f"Skills loaded: {len(skills.list_skills())}"
    )
    await message.answer(response)

@dp.message()
async def main_handler(message: Message) -> None:
    if message.text:
        response = await consciousness.process_input(message.text)
        await message.answer(response)

async def main() -> None:
    # Start subconscious reflection in the background
    asyncio.create_task(subconsciousness.run_reflection_loop())

    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    if BOT_TOKEN != "dummy_token":
        asyncio.run(main())
    else:
        logging.info("Dummy token detected. Running in dry-run mode for integration check.")
        # Minimal async run to check initialization
        async def dry_run():
            logging.info("Initialization successful. Cognitive modules ready.")
        asyncio.run(dry_run())
