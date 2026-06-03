import asyncio
import logging
import os
import sys
from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware, Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

# Initialize Bot and Dispatcher
# In a real setup, BOT_TOKEN would be loaded from an environment variable.
BOT_TOKEN = os.getenv("BOT_TOKEN", "dummy_token")


class WhitelistMiddleware(BaseMiddleware):
    def __init__(self):
        super().__init__()
        allowed_ids_str = os.getenv("ALLOWED_USER_IDS", "")
        self.allowed_user_ids = []
        if allowed_ids_str:
            try:
                self.allowed_user_ids = [int(id_str.strip()) for id_str in allowed_ids_str.split(",") if id_str.strip()]
            except ValueError:
                logging.error("Invalid ALLOWED_USER_IDS environment variable. Must be comma-separated integers.")

    async def __call__(
        self,
        handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: Dict[str, Any]
    ) -> Any:
        if self.allowed_user_ids:
            if event.from_user is None or event.from_user.id not in self.allowed_user_ids:
                logging.warning(f"Unauthorized access attempt by user: {event.from_user.id if event.from_user else 'Unknown'}")
                return # Block the event

        return await handler(event, data)


dp = Dispatcher()
dp.message.middleware(WhitelistMiddleware())

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
