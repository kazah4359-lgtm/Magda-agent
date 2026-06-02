import asyncio
import logging
import os
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

# Initialize Bot and Dispatcher
# In a real setup, BOT_TOKEN would be loaded from an environment variable.
BOT_TOKEN = os.getenv("BOT_TOKEN", "dummy_token")

dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.answer(f"Hello, {message.from_user.full_name}! I am Magda, your AGI agent.")

async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # And the run events dispatching
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    # Only run the bot if a real token is provided, otherwise just exit for tests
    if BOT_TOKEN != "dummy_token":
        asyncio.run(main())
    else:
        logging.info("Dummy token detected. Exiting gracefully without connecting to Telegram.")
