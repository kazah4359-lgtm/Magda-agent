import asyncio
import logging
import os
import sys
from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware, Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, ErrorEvent

from magda_agent.llm_client import LLMClient
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.memory.storage import MemorySystem
from magda_agent.skills import initialize_skills
from magda_agent.consciousness.core import Consciousness
from magda_agent.subconsciousness.reflection import Subconsciousness

# Initialize Components
llm_client = LLMClient()
emotional_engine = EmotionalEngine()
memory_system = MemorySystem()
skill_registry = initialize_skills()

consciousness = Consciousness(
    llm=llm_client,
    emotions=emotional_engine,
    memory=memory_system,
    skills=skill_registry
)

subconsciousness = Subconsciousness(
    llm=llm_client,
    emotions=emotional_engine,
    memory=memory_system,
    interval=300 # Reflect every 5 minutes in production
)

# Initialize Bot and Dispatcher
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
    logging.error(f"Update: {event.update}\nException: {event.exception}", exc_info=True)

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Hello, {message.from_user.full_name}! I am Magda, your AGI agent. I have a mind of my own now.")

@dp.message(Command("state"))
async def command_state_handler(message: Message) -> None:
    """Returns the internal state of the agent."""
    state_info = consciousness.get_internal_state()
    await message.answer(f"<b>My Internal State:</b>\n<pre>{state_info}</pre>")

@dp.message()
async def main_message_handler(message: Message) -> None:
    """Processes all incoming messages through Magda's Consciousness."""
    if not message.text:
        return

    # Show typing status to user
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    # Process through consciousness
    response = await consciousness.process_input(message.text)
    await message.answer(response)

async def main() -> None:
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # Start Subconsciousness in background
    asyncio.create_task(subconsciousness.start())

    # Start Polling
    logging.info("Magda Agent is starting...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    if BOT_TOKEN != "dummy_token":
        try:
            asyncio.run(main())
        except (KeyboardInterrupt, SystemExit):
            logging.info("Magda Agent stopped.")
    else:
        logging.info("Dummy token detected. Running in test mode.")
        # Minimal verification that modules are linked
        print(consciousness.get_internal_state())
