from typing import Any, Callable, Dict, Optional
import asyncio
from magda_agent.gateway.router import UnifiedMessage

class LocalFirstGateway:
    """
    Local-first gateway as a control plane for channel routing.
    Routes incoming channel messages to the agent core safely.
    """
    def __init__(self) -> None:
        self._channels: Dict[str, Any] = {}
        self._message_handler: Optional[Callable[[UnifiedMessage], Any]] = None

    def register_channel(self, channel_id: str, channel: Any) -> None:
        """Register a channel with the gateway control plane."""
        self._channels[channel_id] = channel

    def get_channel(self, channel_id: str) -> Any:
        """Retrieve a registered channel by its ID."""
        return self._channels.get(channel_id)

    def set_message_handler(self, handler: Callable[[UnifiedMessage], Any]) -> None:
        """Set the main handler that processes unified messages."""
        self._message_handler = handler

    async def route_message(self, message: UnifiedMessage) -> Any:
        """Route an incoming message to the registered handler safely."""
        if self._message_handler is None:
            raise RuntimeError("No message handler registered with LocalFirstGateway")
        if asyncio.iscoroutinefunction(self._message_handler):
            return await self._message_handler(message)
        return self._message_handler(message)
