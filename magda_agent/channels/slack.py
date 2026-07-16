from typing import Any, Dict, Optional
from magda_agent.channels.base import ChannelAdapter
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage

class SlackAdapter(ChannelAdapter):
    """Adapter for Slack platform to integrate with the ChannelHub."""

    def __init__(self, gateway: GatewayRouter) -> None:
        """Initialize the Slack platform channel adapter.

        Args:
            gateway (GatewayRouter): The unified routing gateway.
        """
        super().__init__("slack", gateway)

    async def receive(self, raw_data: Any) -> Any:
        """Process an incoming raw Slack message and route it as a UnifiedMessage.

        Args:
            raw_data (Any): The raw data representing the incoming message.
                            Expected to be a dictionary or object with body/text/user attributes.

        Returns:
            Any: The response from the routed gateway message handler.
        """
        text: str = ""
        user_id: str = ""

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
        """Send an outgoing message via the Slack platform.

        Args:
            recipient_id (str): The recipient's Slack user ID.
            text (str): The text message to send.
            metadata (Optional[Dict[str, Any]]): Additional metadata for sending.

        Returns:
            Any: A mock status or confirmation string.
        """
        return f"Slack sent to {recipient_id}: {text}"
