"""
Discord Bridge Module.

This module implements a bridge for Discord integration, allowing the agent
to receive and respond to messages on Discord.
"""
import asyncio
import logging
from typing import Callable, Awaitable, Any, Optional
import discord

logger = logging.getLogger(__name__)

class DiscordBridge:
    """Bridge for Discord integration."""

    def __init__(self, token: str, agent_callback: Callable[[str, Optional[int]], Awaitable[str]]):
        """
        Initialize the Discord Bridge.

        Args:
            token: The Discord bot token.
            agent_callback: An async callback function taking (user_input, user_id) and returning a response.
        """
        self.token = token
        self.agent_callback = agent_callback
        self.is_running = False

        # Configure intents
        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_message(message: discord.Message) -> None:
            # Ignore messages from the bot itself
            if message.author == self.client.user:
                return

            response = await self.on_message(str(message.author.id), message.content)
            if response:
                await message.channel.send(response)

        @self.client.event
        async def on_ready() -> None:
            logger.info(f"Discord bridge connected as {self.client.user}")

    async def start(self) -> None:
        """Start the Discord bridge."""
        if self.token == "dummy" or not self.token:
            logger.warning("Discord token not provided or is 'dummy'. Discord bridge running in mock mode.")
            self.is_running = True
            return

        logger.info("Starting Discord bridge...")
        self.is_running = True
        try:
            await self.client.start(self.token)
        except Exception as e:
            logger.error(f"Failed to start Discord client: {e}")
            self.is_running = False

    async def stop(self) -> None:
        """Stop the Discord bridge."""
        logger.info("Stopping Discord bridge...")
        self.is_running = False
        if self.token != "dummy" and self.token:
            await self.client.close()

    async def on_message(self, user_id: str, content: str) -> Optional[str]:
        """
        Handle incoming message from Discord.

        Args:
            user_id: The ID of the user who sent the message.
            content: The content of the message.

        Returns:
            The response from the agent, or None if the bridge is not running.
        """
        if not self.is_running:
            logger.warning("Received message while bridge is stopped.")
            return None

        logger.info(f"Received message from {user_id}: {content}")
        try:
            uid = int(user_id) if user_id.isdigit() else None
            response = await self.agent_callback(content, uid)
            return response
        except Exception as e:
            logger.error(f"Error in agent callback: {e}")
            return "Error processing request."

    async def send_message(self, user_id: str, message: str) -> None:
        """
        Send a message to a user on Discord. Used for cross-platform integration.

        Args:
            user_id: The Discord user ID.
            message: The message to send.
        """
        if not self.is_running:
            logger.warning("Cannot send message, bridge is stopped.")
            return

        logger.info(f"Sending message to {user_id} on Discord.")
        if self.token == "dummy" or not self.token:
            # Mock sending
            return

        try:
            uid = int(user_id)
            user = await self.client.fetch_user(uid)
            if user:
                await user.send(message)
            else:
                logger.error(f"User {user_id} not found on Discord.")
        except Exception as e:
            logger.error(f"Error sending message to {user_id} on Discord: {e}")
