from typing import Any, Dict, Optional
from magda_agent.channels.base import ChannelAdapter
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage
from magda_agent.channels.hub import ChannelHub
from magda_agent.channels.telegram import TelegramAdapter
from magda_agent.channels.discord import DiscordAdapter

class SlackAdapter(ChannelAdapter):
    """Adapter for Slack platform."""

    def __init__(self, gateway: GatewayRouter):
        super().__init__("slack", gateway)

    async def receive(self, raw_data: Any) -> Any:
        """Process incoming Slack message.

        Args:
            raw_data (Any): The raw Slack event dictionary.

        Returns:
            Any: The response from the gateway.
        """
        text = ""
        user_id = ""

        if isinstance(raw_data, dict):
            text = raw_data.get("text", "")
            user_id = str(raw_data.get("user", ""))
        else:
            text = getattr(raw_data, "text", "")
            user_id = str(getattr(raw_data, "user", ""))

        msg = UnifiedMessage(
            channel=self.channel_id,
            text=text,
            user_id=user_id,
            metadata={"raw": raw_data}
        )
        return await self.gateway.route_message(msg)

    async def send(self, recipient_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Send message via Slack.

        Args:
            recipient_id (str): Slack user ID.
            text (str): Message text.
            metadata (Optional[Dict[str, Any]]): Metadata.

        Returns:
            Any: Mocked sending confirmation.
        """
        return f"Slack sent to {recipient_id}: {text}"


class MultiPlatformDispatcher:
    """Orchestrator class that sets up and dispatches to multiple channels."""

    def __init__(self, gateway: GatewayRouter, bot_token: Optional[str] = None):
        """Initialize multiple platform adapters.

        Args:
            gateway (GatewayRouter): Gateway router.
            bot_token (Optional[str]): Bot token for Telegram.
        """
        self.gateway = gateway
        self.hub = ChannelHub()

        self.telegram = TelegramAdapter(gateway, bot_token=bot_token)
        self.discord = DiscordAdapter(gateway)
        self.slack = SlackAdapter(gateway)

        self.hub.register_adapter(self.telegram)
        self.hub.register_adapter(self.discord)
        self.hub.register_adapter(self.slack)

    async def dispatch(self, channel_id: str, recipient_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Dispatch a message to the specified channel using the ChannelHub.

        Args:
            channel_id (str): Target channel (e.g., 'telegram', 'discord', 'slack').
            recipient_id (str): Recipient ID.
            text (str): Message text.
            metadata (Optional[Dict[str, Any]]): Additional metadata.

        Returns:
            Any: Result of the send operation.
        """
        return await self.hub.send_to_channel(channel_id, recipient_id, text, metadata)
