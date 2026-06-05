from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import asyncio

@dataclass
class UnifiedMessage:
    """Unified internal message format."""
    channel: str
    text: str
    user_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class GatewayRouter:
    """
    Single gateway process that routes messages from multiple channels.
    Routes incoming channel messages to the agent core.
    """
    def __init__(self):
        self._channels: Dict[str, Any] = {}
        self._message_handler: Optional[Callable[[UnifiedMessage], Any]] = None

    def register_channel(self, channel_id: str, channel: Any) -> None:
        """Register a channel with the gateway."""
        self._channels[channel_id] = channel

    def set_message_handler(self, handler: Callable[[UnifiedMessage], Any]) -> None:
        """Set the main handler that processes unified messages."""
        self._message_handler = handler

    async def route_message(self, message: UnifiedMessage) -> Any:
        """Route an incoming message to the registered handler."""
        if self._message_handler is None:
            raise RuntimeError("No message handler registered with GatewayRouter")
        if asyncio.iscoroutinefunction(self._message_handler):
            return await self._message_handler(message)
        return self._message_handler(message)
