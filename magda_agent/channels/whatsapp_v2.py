from typing import Any, Dict, Optional
from magda_agent.channels.base import ChannelAdapter
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage

class WhatsAppAdapterV2(ChannelAdapter):
    """Adapter for WhatsApp (v2) platform supporting unified gateway routing."""

    def __init__(self, gateway: GatewayRouter) -> None:
        """Initialize the WhatsApp v2 channel adapter.

        Args:
            gateway (GatewayRouter): The unified routing gateway.
        """
        super().__init__("whatsapp_v2", gateway)

    async def receive(self, raw_data: Any) -> Any:
        """Process an incoming raw WhatsApp message and route it.

        Args:
            raw_data (Any): The raw data representing the incoming message.
                            Expected to be a dictionary or object with body/text/from/sender attributes.

        Returns:
            Any: The response from the routed gateway message handler.
        """
        text: str = ""
        user_id: str = ""

        if isinstance(raw_data, dict):
            text = raw_data.get("body", raw_data.get("text", ""))
            user_id = str(raw_data.get("from", raw_data.get("sender_id", "")))
        else:
            text = getattr(raw_data, "body", getattr(raw_data, "text", ""))
            user_id = str(getattr(raw_data, "from", getattr(raw_data, "sender_id", "")))

        msg = UnifiedMessage(
            channel=self.channel_id,
            text=text,
            user_id=user_id,
            metadata={"raw": raw_data}
        )
        return await self.gateway.route_message(msg)

    async def send(self, recipient_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Send an outgoing message via the WhatsApp platform.

        Args:
            recipient_id (str): The recipient's ID or phone number.
            text (str): The text message to send.
            metadata (Optional[Dict[str, Any]]): Additional metadata for sending.

        Returns:
            Any: A mock status or confirmation string.
        """
        # Simulated or mocked WhatsApp API response
        return f"WhatsApp V2 sent to {recipient_id}: {text}"
