"""
Omnichannel communication skill.
Supports synchronous compatibility for the skill registry and a first-class async
API for real channel adapters. Gateway sends are awaited and errors are returned
instead of being hidden in fire-and-forget tasks.
"""
import asyncio
import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from magda_agent.gateway.router import GatewayRouter

_BACKGROUND_LOOP: Optional[asyncio.AbstractEventLoop] = None
_BACKGROUND_THREAD: Optional[threading.Thread] = None
_BACKGROUND_LOCK = threading.Lock()


def _get_background_loop() -> asyncio.AbstractEventLoop:
    global _BACKGROUND_LOOP, _BACKGROUND_THREAD
    with _BACKGROUND_LOCK:
        if _BACKGROUND_LOOP and _BACKGROUND_LOOP.is_running():
            return _BACKGROUND_LOOP

        loop = asyncio.new_event_loop()

        def run_loop() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=run_loop, name="magda-omnichannel-loop", daemon=True)
        thread.start()
        _BACKGROUND_LOOP = loop
        _BACKGROUND_THREAD = thread
        return loop


async def send_message_async(platform: str, recipient: str, message: str, gateway: Optional["GatewayRouter"] = None) -> str:
    """Send a message and await the real channel result when a gateway is provided."""
    platform_key = platform.lower()

    if gateway:
        channel = gateway.get_channel(platform_key)
        if not channel:
            return f"Error: Platform '{platform}' is not supported."
        try:
            result = await channel.send(recipient, message)
            return str(result)
        except Exception as e:
            return f"Error: Failed to send {platform_key} message to {recipient}: {e}"

    if platform_key == "telegram":
        return f"Success: Telegram message sent to {recipient}: {message}"
    if platform_key == "whatsapp":
        return f"Success: WhatsApp message sent to {recipient}: {message}"
    if platform_key == "email":
        return f"Success: Email sent to {recipient}: {message}"
    return f"Error: Platform '{platform}' is not supported."


def send_message(platform: str, recipient: str, message: str, gateway: Optional["GatewayRouter"] = None) -> str:
    """
    Synchronous compatibility wrapper for SkillRegistry.

    If called from an async event loop directly, callers should use
    send_message_async() so delivery can be awaited without blocking the loop.
    From synchronous contexts, the coroutine is executed on one dedicated
    background loop and this function waits for the real result.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = _get_background_loop()
        future = asyncio.run_coroutine_threadsafe(
            send_message_async(platform, recipient, message, gateway=gateway),
            loop,
        )
        return future.result(timeout=30)

    return "Error: send_message called from an active event loop; use send_message_async instead."
