from typing import Any, Dict, Optional
from magda_agent.channels.base import ChannelAdapter
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage

class SlackAdapter(ChannelAdapter):
    """Adapter for Slack platform."""

    def __init__(self, gateway: GatewayRouter):
        """Initialize Slack Adapter."""
        super().__init__("slack", gateway)

    async def receive(self, raw_data: Any) -> Any:
        """Process incoming Slack message."""
        text = raw_data.get("text", "") if isinstance(raw_data, dict) else getattr(raw_data, "text", "")
        user_id = raw_data.get("user", "") if isinstance(raw_data, dict) else getattr(raw_data, "user", "")

        msg = UnifiedMessage(
            channel=self.channel_id,
            text=text,
            user_id=user_id,
            metadata={"raw": raw_data}
        )
        return await self.gateway.route_message(msg)

    async def send(self, recipient_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Send message via Slack."""
        return f"Slack sent to {recipient_id}: {text}"


class WhatsAppAdapter(ChannelAdapter):
    """Adapter for WhatsApp platform."""

    def __init__(self, gateway: GatewayRouter):
        """Initialize WhatsApp Adapter."""
        super().__init__("whatsapp", gateway)

    async def receive(self, raw_data: Any) -> Any:
        """Process incoming WhatsApp message."""
        text = raw_data.get("body", "") if isinstance(raw_data, dict) else getattr(raw_data, "body", "")
        user_id = raw_data.get("from", "") if isinstance(raw_data, dict) else getattr(raw_data, "from", "")

        msg = UnifiedMessage(
            channel=self.channel_id,
            text=text,
            user_id=user_id,
            metadata={"raw": raw_data}
        )
        return await self.gateway.route_message(msg)

    async def send(self, recipient_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Send message via WhatsApp."""
        return f"WhatsApp sent to {recipient_id}: {text}"


class SignalAdapter(ChannelAdapter):
    """Adapter for Signal platform."""

    def __init__(self, gateway: GatewayRouter):
        """Initialize Signal Adapter."""
        super().__init__("signal", gateway)

    async def receive(self, raw_data: Any) -> Any:
        """Process incoming Signal message."""
        text = raw_data.get("message", "") if isinstance(raw_data, dict) else getattr(raw_data, "message", "")
        user_id = raw_data.get("source", "") if isinstance(raw_data, dict) else getattr(raw_data, "source", "")

        msg = UnifiedMessage(
            channel=self.channel_id,
            text=text,
            user_id=user_id,
            metadata={"raw": raw_data}
        )
        return await self.gateway.route_message(msg)

    async def send(self, recipient_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Send message via Signal."""
        return f"Signal sent to {recipient_id}: {text}"


class CLIAdapter(ChannelAdapter):
    """Adapter for CLI interactions."""

    def __init__(self, gateway: GatewayRouter):
        """Initialize CLI Adapter."""
        super().__init__("cli", gateway)

    async def receive(self, raw_data: Any) -> Any:
        """Process incoming CLI message."""
        text = raw_data.get("input", "") if isinstance(raw_data, dict) else str(raw_data)
        user_id = "local_user"

        msg = UnifiedMessage(
            channel=self.channel_id,
            text=text,
            user_id=user_id,
            metadata={"raw": raw_data}
        )
        return await self.gateway.route_message(msg)

    async def send(self, recipient_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Send message via CLI."""
        return f"CLI output for {recipient_id}: {text}"
