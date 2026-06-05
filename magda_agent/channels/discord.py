from typing import Any, Dict
from magda_agent.channels.base import BaseChannel
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage

class DiscordChannel(BaseChannel):
    """Adapter for Discord platform."""

    def __init__(self, gateway: GatewayRouter):
        super().__init__("discord", gateway)

    async def receive(self, raw_data: Any) -> Any:
        """Process incoming Discord message."""
        # Assuming raw_data has content and author.id
        text = getattr(raw_data, "content", "")
        if not text and isinstance(raw_data, dict):
            text = raw_data.get("content", "")

        author = getattr(raw_data, "author", None)
        user_id = str(author.id) if author and getattr(author, "id", None) else ""
        if not user_id and isinstance(raw_data, dict):
            user_id = str(raw_data.get("author_id", ""))

        msg = UnifiedMessage(
            channel=self.channel_id,
            text=text,
            user_id=user_id,
            metadata={"raw": raw_data}
        )
        return await self.gateway.route_message(msg)

    async def send(self, recipient_id: str, text: str, metadata: Dict[str, Any] = None) -> Any:
        """Send message via Discord."""
        # This would use discord.py Client in a real implementation
        return f"Discord sent to {recipient_id}: {text}"
