from typing import Any, Dict, Optional
from aiogram import Bot
from aiogram.types import Message, User
from aiogram.exceptions import TelegramAPIError

from magda_agent.channels.base import ChannelAdapter
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage

class TelegramAdapter(ChannelAdapter):
    """Adapter for Telegram platform."""

    def __init__(self, gateway: GatewayRouter, bot_token: Optional[str] = None):
        super().__init__("telegram", gateway)
        self.bot_token = bot_token
        self._bot: Optional[Bot] = None
        if self.bot_token:
            self._bot = Bot(token=self.bot_token)

    async def receive(self, raw_data: Any) -> Any:
        """Process incoming Telegram message.

        Args:
            raw_data (Any): The incoming aiogram Message object or a dictionary fallback.

        Returns:
            Any: The response from the gateway router.
        """
        text = ""
        user_id = ""

        if isinstance(raw_data, Message):
            text = raw_data.text or ""
            if raw_data.from_user:
                user_id = str(raw_data.from_user.id)
        else:
            text = getattr(raw_data, "text", "")
            if not text and isinstance(raw_data, dict):
                text = raw_data.get("text", "")

            user = getattr(raw_data, "from_user", None)
            if user and getattr(user, "id", None):
                user_id = str(user.id)
            elif not user_id and isinstance(raw_data, dict):
                user_id = str(raw_data.get("user_id", ""))

        msg = UnifiedMessage(
            channel=self.channel_id,
            text=text,
            user_id=user_id,
            metadata={"raw": raw_data}
        )
        return await self.gateway.route_message(msg)

    async def send(self, recipient_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Send message via Telegram.

        Args:
            recipient_id (str): The Telegram user ID to send the message to.
            text (str): The message text.
            metadata (Optional[Dict[str, Any]]): Additional metadata for sending.

        Returns:
            Any: The sent message object or a fallback string on error/missing token.
        """
        bot = self._bot
        # Allow passing an explicit Bot instance in metadata for testing or dynamic bots
        if metadata and "bot" in metadata:
            bot = metadata["bot"]

        if not bot:
            return f"Telegram sent to {recipient_id}: {text} (Mock/No Token)"

        try:
            return await bot.send_message(chat_id=recipient_id, text=text)
        except TelegramAPIError as e:
            return f"Error sending to {recipient_id}: {str(e)}"
