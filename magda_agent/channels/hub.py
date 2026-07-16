from typing import Any, Dict, Optional
from magda_agent.channels.base import ChannelAdapter

class ChannelHub:
    """
    Centralized messaging hub that abstracts multiple channel adapters.
    Manages a collection of ChannelAdapter instances and allows receiving and sending messages across them.
    """
    def __init__(self, rate_limiter: Optional[Any] = None) -> None:
        """
        Initialize the centralized messaging hub.

        Args:
            rate_limiter (Optional[Any]): Rate limiter instance for outgoing messages.
        """
        self._adapters: Dict[str, ChannelAdapter] = {}
        self.rate_limiter: Optional[Any] = rate_limiter

    def set_rate_limiter(self, rate_limiter: Any) -> None:
        """
        Set or replace the rate limiter for the hub.

        Args:
            rate_limiter (Any): The rate limiter instance.
        """
        self.rate_limiter = rate_limiter

    def register_adapter(self, adapter: ChannelAdapter) -> None:
        """
        Register a ChannelAdapter with the hub.

        Args:
            adapter (ChannelAdapter): The adapter instance to register.
        """
        self._adapters[adapter.channel_id] = adapter

    def get_adapter(self, channel_id: str) -> Optional[ChannelAdapter]:
        """
        Retrieve a registered ChannelAdapter by its ID.

        Args:
            channel_id (str): The ID of the channel.

        Returns:
            Optional[ChannelAdapter]: The adapter if found, otherwise None.
        """
        return self._adapters.get(channel_id)

    async def receive_from_channel(self, channel_id: str, raw_data: Any) -> Any:
        """
        Process incoming messages from a specific channel.

        Args:
            channel_id (str): The ID of the channel.
            raw_data (Any): The raw incoming data.

        Returns:
            Any: The response from the channel adapter's receive method, or None if the adapter is not found.
        """
        adapter = self.get_adapter(channel_id)
        if adapter:
            return await adapter.receive(raw_data)
        return None

    async def send_to_channel(self, channel_id: str, recipient_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """
        Dispatch outgoing messages to a specific channel.

        Args:
            channel_id (str): The ID of the target channel.
            recipient_id (str): The ID of the recipient.
            text (str): The message text.
            metadata (Optional[Dict[str, Any]]): Additional metadata.

        Returns:
            Any: The response from the channel adapter's send method, or an error string if the adapter is not found.
        """
        if self.rate_limiter:
            await self.rate_limiter.acquire(channel_id)

        adapter = self.get_adapter(channel_id)
        if adapter:
            return await adapter.send(recipient_id, text, metadata)
        return f"Error: Channel '{channel_id}' not found."
