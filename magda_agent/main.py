import asyncio
import logging
import os
import sys
from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware, Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, ErrorEvent

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

@dp.errors()
async def error_handler(event: ErrorEvent) -> None:
    """
    Global error handler to catch all asynchronous errors in aiogram handlers.
    """
    logging.error(f"Update: {event.update}\nException: {event.exception}", exc_info=True)


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    await message.answer(f"Hello, {message.from_user.full_name}! I am Magda, your AGI agent.")

@dp.message()
async def process_message_handler(message: Message) -> None:
    """
    This handler receives all other messages and forwards them to consciousness.
    """
    import httpx
    consciousness_url = os.getenv("CONSCIOUSNESS_URL", "http://consciousness:8000")

    if message.text:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{consciousness_url}/process",
                    json={"input_text": message.text},
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                await message.answer(data.get("thoughts", "No thoughts..."))
        except Exception as e:
            logging.error(f"Failed to communicate with consciousness: {e}", exc_info=True)
            await message.answer("My consciousness is currently unreachable.")

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
