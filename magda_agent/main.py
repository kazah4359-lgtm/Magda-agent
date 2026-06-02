import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

import os

from aiogram.types import FSInputFile

from magda_agent.config import settings
from magda_agent.consciousness.llm import Consciousness
from magda_agent.skills.speech import SpeechProcessor

dp = Dispatcher()
consciousness = Consciousness()

# Initialize ML models globally to prevent loading them on every message
speech_processor = SpeechProcessor()

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

@dp.message(F.voice)
async def voice_message_handler(message: Message, bot: Bot) -> None:
    """
    Handler for voice messages.
    Downloads the audio, transcribes it, gets LLM response, synthesizes audio, and replies.
    """
    try:
        voice_ogg_path = f"temp_{message.voice.file_id}.ogg"
        voice_wav_path = f"temp_{message.voice.file_id}.wav"
        response_wav_path = f"response_{message.voice.file_id}.wav"
        response_ogg_path = f"response_{message.voice.file_id}.ogg"

        # 1. Download the voice note
        voice_file_info = await bot.get_file(message.voice.file_id)
        await bot.download_file(voice_file_info.file_path, voice_ogg_path)

        # 2. Convert to WAV for processing (offloaded to thread)
        await asyncio.to_thread(speech_processor.convert_ogg_to_wav, voice_ogg_path, voice_wav_path)

        # 3. Speech to Text (offloaded to thread)
        user_text = await asyncio.to_thread(speech_processor.speech_to_text, voice_wav_path)
        await message.reply(f"🎤 <i>Распознано:</i> {user_text}")

        # 4. Process with Consciousness (LLM)
        response_text = await consciousness.process_message(user_text)

        # 5. Text to Speech (offloaded to thread)
        await asyncio.to_thread(speech_processor.text_to_speech, response_text, response_wav_path)

        # 6. Convert back to OGG for Telegram
        await asyncio.to_thread(speech_processor.convert_wav_to_ogg, response_wav_path, response_ogg_path)

        # 7. Send voice response
        audio_file = FSInputFile(response_ogg_path)
        await message.answer_voice(audio_file)

    except Exception as e:
        await message.answer(f"Произошла ошибка при обработке голосового сообщения: {e}")
    finally:
        # Cleanup temp files safely
        for path in [
            f"temp_{message.voice.file_id}.ogg" if hasattr(message, 'voice') and message.voice else None,
            f"temp_{message.voice.file_id}.wav" if hasattr(message, 'voice') and message.voice else None,
            f"response_{message.voice.file_id}.wav" if hasattr(message, 'voice') and message.voice else None,
            f"response_{message.voice.file_id}.ogg" if hasattr(message, 'voice') and message.voice else None,
        ]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

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
