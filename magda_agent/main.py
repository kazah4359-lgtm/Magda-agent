import asyncio
import logging
import os
import sys
from typing import Any, Awaitable, Callable, Dict
import httpx

from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, ErrorEvent, FSInputFile


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

@dp.message(Command("task"))
async def command_task_handler(message: Message) -> None:
    """Enqueue a long-running autonomous task: /task <goal>."""
    goal = (message.text or "").partition(" ")[2].strip()
    if not goal:
        await message.answer("Usage: <code>/task &lt;goal&gt;</code>\nExample: <code>/task research the top 3 Python web frameworks and summarize tradeoffs</code>")
        return
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            payload = {"goal": goal}
            if message.from_user and message.from_user.id:
                payload["user_id"] = message.from_user.id
            response = await client.post(f"{CONSCIOUSNESS_API_URL}/tasks", json=payload)
            response.raise_for_status()
            task = response.json().get("task", {})
        await message.answer(
            f"Task queued. id: <code>{task.get('id')}</code>\nI'll work on it in the background and post progress here.\n"
            f"Use <code>/tasks</code> to list, <code>/cancel {task.get('id')}</code> to stop."
        )
    except Exception as e:
        logging.error(f"Failed to create task: {e}")
        await message.answer("Error: could not queue the task (Consciousness API unreachable).")


@dp.message(Command("tasks"))
async def command_tasks_handler(message: Message) -> None:
    """List the caller's autonomous tasks."""
    try:
        params = {}
        if message.from_user and message.from_user.id:
            params["user_id"] = message.from_user.id
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{CONSCIOUSNESS_API_URL}/tasks", params=params)
            response.raise_for_status()
            tasks = response.json().get("tasks", [])
        if not tasks:
            await message.answer("No tasks yet. Start one with <code>/task &lt;goal&gt;</code>.")
            return
        lines = ["<b>Your tasks:</b>"]
        for t in tasks[:20]:
            lines.append(
                f"• <code>{t.get('id')}</code> [{t.get('status')}] "
                f"iter {t.get('iterations')}/{t.get('max_iterations')} — {t.get('goal')[:60]}"
            )
        await message.answer("\n".join(lines))
    except Exception as e:
        logging.error(f"Failed to list tasks: {e}")
        await message.answer("Error: could not list tasks.")


async def _task_action(message: Message, action: str) -> None:
    task_id = (message.text or "").partition(" ")[2].strip()
    if not task_id:
        await message.answer(f"Usage: <code>/{action} &lt;task_id&gt;</code>")
        return
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(f"{CONSCIOUSNESS_API_URL}/tasks/{task_id}/{action}")
        if response.status_code == 200:
            await message.answer(f"Task <code>{task_id}</code>: {action} requested.")
        elif response.status_code == 409:
            await message.answer(f"Task <code>{task_id}</code> cannot be {action}d in its current state.")
        else:
            await message.answer(f"Task <code>{task_id}</code> not found.")
    except Exception as e:
        logging.error(f"Failed to {action} task: {e}")
        await message.answer(f"Error: could not {action} the task.")


@dp.message(Command("cancel"))
async def command_cancel_handler(message: Message) -> None:
    await _task_action(message, "cancel")


@dp.message(Command("pause"))
async def command_pause_handler(message: Message) -> None:
    await _task_action(message, "pause")


@dp.message(Command("resume"))
async def command_resume_handler(message: Message) -> None:
    await _task_action(message, "resume")


async def task_progress_poller(bot: Bot, interval: float = 5.0) -> None:
    """Polls the Consciousness API for new task progress and pushes it to users."""
    delivered: Dict[str, int] = {}
    while True:
        await asyncio.sleep(interval)
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(f"{CONSCIOUSNESS_API_URL}/tasks")
                resp.raise_for_status()
                tasks = resp.json().get("tasks", [])
                for t in tasks:
                    task_id = t.get("id")
                    user_id = t.get("user_id")
                    if not task_id or not user_id:
                        continue
                    seen = delivered.get(task_id, 0)
                    count = t.get("progress_count", 0)
                    if count <= seen:
                        continue
                    detail = await client.get(f"{CONSCIOUSNESS_API_URL}/tasks/{task_id}", params={"since": seen})
                    detail.raise_for_status()
                    new_entries = detail.json().get("progress", [])
                    for entry in new_entries:
                        text = f"[{task_id}] {entry.get('event', 'info')}: {entry.get('message', '')}"
                        try:
                            await bot.send_message(chat_id=user_id, text=text[:4000])
                        except Exception as send_err:
                            logging.warning(f"Could not deliver progress to {user_id}: {send_err}")
                    delivered[task_id] = count
        except Exception as e:
            logging.debug(f"Progress poller iteration failed: {e}")


@dp.message(F.voice)
async def voice_message_handler(message: Message) -> None:
    """Processes incoming voice messages."""
    await message.bot.send_chat_action(chat_id=message.chat.id, action="record_voice")

    try:
        from magda_agent.speech.processor import SpeechProcessor
        speech_processor = SpeechProcessor()
        # Download voice message
        file_id = message.voice.file_id
        file = await message.bot.get_file(file_id)
        file_path = file.file_path

        # Save locally
        local_ogg = f"/tmp/{file_id}.ogg"
        await message.bot.download_file(file_path, local_ogg)

        # STT
        text = await speech_processor.stt(local_ogg)
        if os.path.exists(local_ogg):
            os.remove(local_ogg)

        # Send to Consciousness API
        async with httpx.AsyncClient(timeout=180.0) as client:
            payload = {"text": text}
            if message.from_user and message.from_user.id:
                payload["user_id"] = message.from_user.id

            response = await client.post(
                f"{CONSCIOUSNESS_API_URL}/process",
                json=payload
            )
            response.raise_for_status()
            resp_text = response.json().get("response", "No response from API.")

        # TTS
        out_ogg = f"/tmp/out_{file_id}.ogg"
        await speech_processor.tts(resp_text, out_ogg)

        # Send voice back
        voice_file = FSInputFile(out_ogg)
        await message.answer_voice(voice_file)

        if os.path.exists(out_ogg):
            os.remove(out_ogg)

    except Exception as e:
        logging.error(f"Failed to process voice input: {e}")
        await message.answer("Error processing your voice message.")

@dp.message()
async def main_message_handler(message: Message) -> None:
    """Processes all incoming messages through Magda's Consciousness microservice."""
    if not message.text:
        return

    # Show typing status to user
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            payload = {"text": message.text}
            if message.from_user and message.from_user.id:
                payload["user_id"] = message.from_user.id

            response = await client.post(
                f"{CONSCIOUSNESS_API_URL}/process",
                json=payload
            )
            response.raise_for_status()
            resp_text = response.json().get("response", "No response from API.")
            await message.answer(resp_text)
    except Exception as e:
        logging.error(f"Failed to process input: {e}")
        await message.answer("Error: Consciousness API is unreachable or returned an error.")

async def main() -> None:
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # Start background progress poller for long-running autonomous tasks.
    poller = asyncio.create_task(task_progress_poller(bot))

    # Start Polling
    logging.info("Magda Agent is starting...")
    try:
        await dp.start_polling(bot)
    finally:
        poller.cancel()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    if BOT_TOKEN != "dummy_token":
        try:
            asyncio.run(main())
        except (KeyboardInterrupt, SystemExit):
            logging.info("Magda Agent stopped.")
    else:
        logging.info("Dummy token detected. Running in test mode.")
