import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

from magda_agent.config import settings
from magda_agent.consciousness.llm import Consciousness

dp = Dispatcher()
consciousness = Consciousness()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.answer(f"Привет, {message.from_user.full_name}! Я Магда, ваш AGI агент.")

@dp.message(F.text)
async def message_handler(message: Message) -> None:
    """
    Handler for all text messages. Routes the message to the consciousness module.
    """
    if message.text:
        response_text = await consciousness.process_message(message.text)
        await message.answer(response_text)

async def main() -> None:
    bot_token = settings.bot_token.get_secret_value()
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # And the run events dispatching
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    # Only run the bot if a real token is provided, otherwise just exit for tests
    if settings.bot_token.get_secret_value() != "dummy_token":
        asyncio.run(main())
    else:
        logging.info("Dummy token detected. Exiting gracefully without connecting to Telegram.")
