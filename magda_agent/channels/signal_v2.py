from typing import Any, Dict, Optional
from magda_agent.channels.base import ChannelAdapter
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage

class SignalAdapterV2(ChannelAdapter):
    """Adapter for Signal platform supporting unified gateway routing."""

    def __init__(self, gateway: GatewayRouter) -> None:
        """Initialize the Signal v2 channel adapter.

        Args:
            gateway (GatewayRouter): The unified routing gateway.
        """
        super().__init__("signal_v2", gateway)

    async def receive(self, raw_data: Any) -> Any:
        """Process an incoming raw Signal message and route it.

        Args:
            raw_data (Any): The raw data representing the incoming message.
                            Expected to have body and source (sender).

        Returns:
            Any: The response from the routed gateway message handler.
        """
        text: str = ""
        user_id: str = ""

        if isinstance(raw_data, dict):
            text = raw_data.get("body", "")
            user_id = str(raw_data.get("source", ""))
        else:
            text = getattr(raw_data, "body", "")
            user_id = str(getattr(raw_data, "source", ""))

        msg = UnifiedMessage(
            channel=self.channel_id,
            text=text,
            user_id=user_id,
            metadata={"raw": raw_data}
        )
        return await self.gateway.route_message(msg)

    async def send(self, recipient_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Send an outgoing message via the Signal platform.

        Args:
            recipient_id (str): The recipient's Signal ID or phone number.
            text (str): The text message to send.
            metadata (Optional[Dict[str, Any]]): Additional metadata for sending.

        Returns:
            Any: A mock status or confirmation string.
        """
        # Simulated or mocked Signal API response
        return f"Signal V2 sent to {recipient_id}: {text}"
