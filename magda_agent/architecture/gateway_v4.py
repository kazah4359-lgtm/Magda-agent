from typing import Any, Callable, Dict, List, Optional, Awaitable
import asyncio
import logging
from magda_agent.gateway.router import UnifiedMessage

logger = logging.getLogger(__name__)

# Type definition for middleware
MiddlewareType = Callable[[UnifiedMessage, Callable[[UnifiedMessage], Awaitable[Any]]], Awaitable[Any]]

class UnifiedRoutingLayer:
    """
    Unified routing layer that manages message flow from channels to the agent core.
    Supports middleware for cross-cutting concerns like logging, safety, and telemetry.
    """
    def __init__(self, message_handler: Callable[[UnifiedMessage], Awaitable[Any]]):
        """
        Initialize the routing layer.

        Args:
            message_handler (Callable[[UnifiedMessage], Awaitable[Any]]): The final handler for messages.
        """
        self._handler = message_handler
        self._middlewares: List[MiddlewareType] = []

    def use(self, middleware: MiddlewareType) -> None:
        """
        Add a middleware to the routing layer.

        Args:
            middleware (MiddlewareType): The middleware function to add.
        """
        self._middlewares.append(middleware)

    async def handle(self, message: UnifiedMessage) -> Any:
        """
        Handle a message by passing it through the middleware chain and then to the handler.

        Args:
            message (UnifiedMessage): The message to route.

        Returns:
            Any: The result of processing the message.
        """
        async def _execute_chain(index: int, msg: UnifiedMessage) -> Any:
            if index < len(self._middlewares):
                middleware = self._middlewares[index]
                async def next_call(m: UnifiedMessage) -> Any:
                    return await _execute_chain(index + 1, m)
                return await middleware(msg, next_call)

            return await self._handler(msg)

        return await _execute_chain(0, message)

class LocalFirstGatewayV4:
    """
    OpenClaw-inspired local-first gateway. Acts as the central control plane
    for multi-channel message routing and event processing.
    """
    def __init__(self, message_handler: Callable[[UnifiedMessage], Awaitable[Any]]):
        """
        Initialize the gateway.

        Args:
            message_handler (Callable[[UnifiedMessage], Awaitable[Any]]): The main message handler.
        """
        self._router = UnifiedRoutingLayer(message_handler)
        self._channels: Dict[str, Any] = {}

    def register_channel(self, channel_id: str, channel_adapter: Any) -> None:
        """
        Register a channel adapter with the gateway.

        Args:
            channel_id (str): Unique identifier for the channel.
            channel_adapter (Any): The adapter instance for the channel.
        """
        self._channels[channel_id] = channel_adapter
        logger.info(f"Channel {channel_id} registered with LocalFirstGatewayV4")

    def use(self, middleware: MiddlewareType) -> None:
        """
        Add a middleware to the gateway's routing layer.

        Args:
            middleware (MiddlewareType): The middleware to add.
        """
        self._router.use(middleware)

    async def route(self, message: UnifiedMessage) -> Any:
        """
        Route an incoming unified message through the gateway.

        Args:
            message (UnifiedMessage): The incoming message.

        Returns:
            Any: The result of routing the message.
        """
        return await self._router.handle(message)

    def get_channel(self, channel_id: str) -> Optional[Any]:
        """
        Get a registered channel by ID.

        Args:
            channel_id (str): The ID of the channel to retrieve.

        Returns:
            Optional[Any]: The channel adapter if found, otherwise None.
        """
        return self._channels.get(channel_id)
