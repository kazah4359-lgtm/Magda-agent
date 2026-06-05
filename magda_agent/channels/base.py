from abc import ABC, abstractmethod
from typing import Any, Dict
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage

class BaseChannel(ABC):
    """Base class for all communication channels."""

    def __init__(self, channel_id: str, gateway: GatewayRouter):
        self.channel_id = channel_id
        self.gateway = gateway
        self.gateway.register_channel(channel_id, self)

    @abstractmethod
    async def receive(self, raw_data: Any) -> Any:
        """Process raw incoming data and route it as a UnifiedMessage."""
        pass

    @abstractmethod
    async def send(self, recipient_id: str, text: str, metadata: Dict[str, Any] = None) -> Any:
        """Send a message out through this channel."""
        pass
