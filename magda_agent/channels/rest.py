from typing import Any, Dict
from magda_agent.channels.base import ChannelAdapter
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage

class RestAdapter(ChannelAdapter):
    """Adapter for REST API."""

    def __init__(self, gateway: GatewayRouter):
        super().__init__("rest", gateway)

    async def receive(self, raw_data: Dict[str, Any]) -> Any:
        """Process incoming REST request payload."""
        text = raw_data.get("text", "")
        user_id = str(raw_data.get("user_id", ""))

        msg = UnifiedMessage(
            channel=self.channel_id,
            text=text,
            user_id=user_id,
            metadata={"raw": raw_data}
        )
        return await self.gateway.route_message(msg)

    async def send(self, recipient_id: str, text: str, metadata: Dict[str, Any] = None) -> Any:
        """Send message via REST API."""
        # In REST context, responses are usually returned synchronously,
        # but this could also be a webhook callback
        return {"status": "success", "recipient": recipient_id, "text": text}
