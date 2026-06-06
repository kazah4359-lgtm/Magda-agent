from typing import Any, Dict
from magda_agent.channels.base import ChannelAdapter
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage

class TelegramAdapter(ChannelAdapter):
    """Adapter for Telegram platform."""

    def __init__(self, gateway: GatewayRouter):
        super().__init__("telegram", gateway)

    async def receive(self, raw_data: Any) -> Any:
        """Process incoming Telegram message."""
        # Assuming raw_data is an aiogram Message mock or dict
        text = getattr(raw_data, "text", "")
        if not text and isinstance(raw_data, dict):
            text = raw_data.get("text", "")

        user = getattr(raw_data, "from_user", None)
        user_id = str(user.id) if user and getattr(user, "id", None) else ""
        if not user_id and isinstance(raw_data, dict):
            user_id = str(raw_data.get("user_id", ""))

        msg = UnifiedMessage(
            channel=self.channel_id,
            text=text,
            user_id=user_id,
            metadata={"raw": raw_data}
        )
        return await self.gateway.route_message(msg)

    async def send(self, recipient_id: str, text: str, metadata: Dict[str, Any] = None) -> Any:
        """Send message via Telegram."""
        # This would use aiogram Bot in a real implementation
        return f"Telegram sent to {recipient_id}: {text}"
