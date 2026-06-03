import asyncio
import logging
import os
import sys
from typing import Any, Awaitable, Callable, Dict
import httpx

from aiogram import BaseMiddleware, Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
import tempfile
from pydub import AudioSegment
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, ErrorEvent, FSInputFile
from aiogram import F

from magda_agent.speech import speech_processor

# API URL for Consciousness Microservice
CONSCIOUSNESS_API_URL = os.getenv("CONSCIOUSNESS_API_URL", "http://consciousness:8000")

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
    """Returns the internal state of the agent from the microservice."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{CONSCIOUSNESS_API_URL}/state")
            response.raise_for_status()
            state_info = response.json().get("state", "No state information.")
            await message.answer(f"<b>My Internal State:</b>\n<pre>{state_info}</pre>")
    except Exception as e:
        logging.error(f"Failed to get state: {e}")
        await message.answer("Error: Could not retrieve internal state from Consciousness API.")

@dp.message(F.voice)
async def voice_message_handler(message: Message) -> None:
    """Processes incoming voice messages."""
    await message.bot.send_chat_action(chat_id=message.chat.id, action="record_voice")

    try:
        # Download voice message
        voice = await message.bot.get_file(message.voice.file_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            ogg_path = os.path.join(temp_dir, "input.ogg")
            wav_path = os.path.join(temp_dir, "input.wav")
            out_wav_path = os.path.join(temp_dir, "output.wav")
            out_ogg_path = os.path.join(temp_dir, "output.ogg")

            await message.bot.download_file(voice.file_path, ogg_path)

            # Convert OGG to WAV using pydub in a thread to not block event loop
            def convert_to_wav():
                audio = AudioSegment.from_ogg(ogg_path)
                audio.export(wav_path, format="wav")

            await asyncio.to_thread(convert_to_wav)

            # STT
            text = await asyncio.to_thread(speech_processor.stt, wav_path)

            if not text:
                await message.answer("Sorry, I couldn't understand the audio.")
                return

            # Send to Consciousness API
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{CONSCIOUSNESS_API_URL}/process",
                    json={"text": text}
                )
                response.raise_for_status()
                resp_text = response.json().get("response", "No response from API.")

            # TTS
            success = await asyncio.to_thread(speech_processor.tts, resp_text, out_wav_path)

            if success:
                # Convert WAV back to OGG using pydub
                def convert_to_ogg():
                    audio = AudioSegment.from_wav(out_wav_path)
                    # Telegram requires OPUS codec for voice messages
                    audio.export(out_ogg_path, format="ogg", codec="libopus")

                await asyncio.to_thread(convert_to_ogg)

                # Send voice message
                voice_file = FSInputFile(out_ogg_path)
                await message.answer_voice(voice_file)
            else:
                # Fallback to text
                await message.answer(resp_text)

    except Exception as e:
        logging.error(f"Failed to process voice input: {e}")
        await message.answer("Error processing voice message.")

@dp.message()
async def main_message_handler(message: Message) -> None:
    """Processes all incoming messages through Magda's Consciousness microservice."""
    if not message.text:
        return

    # Show typing status to user
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{CONSCIOUSNESS_API_URL}/process",
                json={"text": message.text}
            )
            response.raise_for_status()
            resp_text = response.json().get("response", "No response from API.")
            await message.answer(resp_text)
    except Exception as e:
        logging.error(f"Failed to process input: {e}")
        await message.answer("Error: Consciousness API is unreachable or returned an error.")

async def main() -> None:
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

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
