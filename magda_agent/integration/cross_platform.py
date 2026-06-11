from typing import Dict, Any, List, Optional
import logging
import asyncio

class CrossPlatformDispatcher:
    """
    Dispatcher to send messages across multiple platforms simultaneously.
    Supports multi-channel reach (e.g., Telegram, Discord, Slack, etc.) from a single agent.
    """
    def __init__(self):
        """Initializes the CrossPlatformDispatcher with an empty registry of platforms."""
        self.platforms: Dict[str, Any] = {}

    def register_platform(self, platform_name: str, client: Any) -> None:
        """
        Registers a platform client.

        Args:
            platform_name: The name of the platform (e.g., 'telegram', 'discord').
            client: The client object that handles message sending for this platform.
                    It must have a `send_message(user_id, message)` async method.
        """
        self.platforms[platform_name] = client
        logging.info(f"Platform registered: {platform_name}")

    async def broadcast(self, user_id: str, message: str) -> Dict[str, str]:
        """
        Broadcasts a message to all registered platforms concurrently.

        Args:
            user_id: The identifier for the user (could be interpreted differently by each platform or a mapped ID).
            message: The message to send.

        Returns:
            A dictionary mapping platform names to their delivery status (e.g., 'success', 'failed').
        """
        if not self.platforms:
            logging.warning("No platforms registered for broadcast.")
            return {}

        results = {}
        async def send_to_platform(name: str, client: Any):
            try:
                # Ensure the client has a send_message method
                if hasattr(client, 'send_message'):
                    await client.send_message(user_id, message)
                    results[name] = 'success'
                else:
                    results[name] = 'failed: send_message method missing'
            except Exception as e:
                logging.error(f"Failed to send message to {name}: {e}")
                results[name] = f'error: {str(e)}'

        tasks = [send_to_platform(name, client) for name, client in self.platforms.items()]
        await asyncio.gather(*tasks)

        logging.info(f"Broadcasted to {len(self.platforms)} platforms. Results: {results}")
        return results
