import asyncio
from typing import Any, Dict
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage

class MockChannel:
    """Mock channel for testing."""
    def __init__(self, channel_id: str):
        self.channel_id = channel_id

    async def process(self, raw_message: Dict[str, Any], router: GatewayRouter) -> Any:
        """Process raw message and route as UnifiedMessage."""
        msg = UnifiedMessage(
            channel=self.channel_id,
            text=raw_message.get("text", ""),
            user_id=raw_message.get("user_id", "unknown"),
            metadata=raw_message
        )
        return await router.route_message(msg)

class TelegramChannel:
    """Telegram channel implementation."""
    def __init__(self, channel_id: str = "telegram"):
        self.channel_id = channel_id

    async def process(self, update: Dict[str, Any], router: GatewayRouter) -> Any:
        """Process telegram update and route."""
        message = update.get("message", {})
        text = message.get("text", "")
        user = message.get("from", {})
        user_id = str(user.get("id", "unknown"))
        msg = UnifiedMessage(
            channel=self.channel_id,
            text=text,
            user_id=user_id,
            metadata={"raw_update": update}
        )
        return await router.route_message(msg)

class DiscordChannel:
    """Discord channel implementation."""
    def __init__(self, channel_id: str = "discord"):
        self.channel_id = channel_id

    async def process(self, event: Dict[str, Any], router: GatewayRouter) -> Any:
        """Process discord event and route."""
        text = event.get("content", "")
        author = event.get("author", {})
        user_id = str(author.get("id", "unknown"))
        msg = UnifiedMessage(
            channel=self.channel_id,
            text=text,
            user_id=user_id,
            metadata={"raw_event": event}
        )
        return await router.route_message(msg)
