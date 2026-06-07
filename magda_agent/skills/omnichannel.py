import logging
import asyncio
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from magda_agent.gateway.router import GatewayRouter

def send_message(platform: str, recipient: str, message: str, gateway: Optional["GatewayRouter"] = None) -> str:
    """
    Sends a message to a recipient on a specified platform.
    Supported platforms (mocked): telegram, whatsapp, email.

    Args:
        platform (str): The communication platform (e.g., 'telegram', 'whatsapp', 'email').
        recipient (str): The recipient's ID, phone number, or email address.
        message (str): The message content to send.
        gateway (GatewayRouter, optional): Gateway router to dispatch messages to specific channels.

    Returns:
        str: A status message indicating the result of the operation.
    """
    platform = platform.lower()
    logging.info(f"Omnichannel: Sending {platform} message to {recipient}: {message}")

    if gateway:
        channel = gateway.get_channel(platform)
        if channel:
            try:
                loop = asyncio.get_running_loop()
                # We are in an event loop (e.g. during an API request or async background task)
                # We cannot block it with run_until_complete. Since this is a sync wrapper
                # around an async call, we schedule it as a fire-and-forget task.
                loop.create_task(channel.send(recipient, message))
                return f"Success (Async): {platform.capitalize()} message dispatched to {recipient}."
            except RuntimeError:
                # No running loop, we are in a synchronous context (e.g. a worker thread)
                return asyncio.run(channel.send(recipient, message))

    if platform == "telegram":
        return f"Success: Telegram message sent to {recipient}."
    elif platform == "whatsapp":
        return f"Success: WhatsApp message sent to {recipient} via mock gateway."
    elif platform == "email":
        return f"Success: Email sent to {recipient} via mock SMTP."
    else:
        return f"Error: Platform '{platform}' is not supported."
