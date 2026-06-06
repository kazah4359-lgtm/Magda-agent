import pytest
import asyncio
from typing import Any, Dict
from magda_agent.channels.base import ChannelAdapter
from magda_agent.channels.discord import DiscordAdapter
from magda_agent.channels.telegram import TelegramAdapter
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage

class MockChannel(ChannelAdapter):
    """A mock channel adapter for testing purposes."""

    def __init__(self, gateway: GatewayRouter) -> None:
        """Initialize the mock channel.

        Args:
            gateway (GatewayRouter): The gateway router to register with.
        """
        super().__init__("mock", gateway)

    async def receive(self, raw_data: Dict[str, Any]) -> Any:
        """Process incoming mock data and route it.

        Args:
            raw_data (Dict[str, Any]): The incoming data.

        Returns:
            Any: The response from the gateway.
        """
        text: str = raw_data.get("text", "")
        user_id: str = raw_data.get("user_id", "")
        msg = UnifiedMessage(
            channel=self.channel_id,
            text=text,
            user_id=user_id,
            metadata={"raw": raw_data}
        )
        return await self.gateway.route_message(msg)

    async def send(self, recipient_id: str, text: str, metadata: Dict[str, Any] = None) -> str:
        """Mock sending a message.

        Args:
            recipient_id (str): The recipient's ID.
            text (str): The message text.
            metadata (Dict[str, Any], optional): Additional metadata.

        Returns:
            str: The mock sent message confirmation.
        """
        return f"Mock sent to {recipient_id}: {text}"

@pytest.fixture
def gateway() -> GatewayRouter:
    """Fixture that provides a GatewayRouter with a mock message handler.

    Returns:
        GatewayRouter: The configured gateway.
    """
    router = GatewayRouter()

    async def mock_handler(msg: UnifiedMessage) -> Dict[str, str]:
        """Mock handler for testing routed messages.

        Args:
            msg (UnifiedMessage): The routed message.

        Returns:
            Dict[str, str]: Handled message details.
        """
        return {"handled_text": msg.text, "handled_user": msg.user_id, "channel": msg.channel}

    router.set_message_handler(mock_handler)
    return router

@pytest.mark.asyncio
async def test_channel_adapter_interface(gateway: GatewayRouter) -> None:
    """Test that the ChannelAdapter interface behaves correctly.

    Args:
        gateway (GatewayRouter): The gateway fixture.
    """
    channel = MockChannel(gateway)
    assert channel.channel_id == "mock"
    assert gateway.get_channel("mock") is channel

    response = await channel.receive({"text": "hello", "user_id": "123"})
    assert response["handled_text"] == "hello"
    assert response["handled_user"] == "123"
    assert response["channel"] == "mock"

    sent = await channel.send("123", "hi")
    assert sent == "Mock sent to 123: hi"

@pytest.mark.asyncio
async def test_discord_adapter(gateway: GatewayRouter) -> None:
    """Test the DiscordAdapter receive logic.

    Args:
        gateway (GatewayRouter): The gateway fixture.
    """
    adapter = DiscordAdapter(gateway)

    class MockAuthor:
        def __init__(self, author_id: str) -> None:
            self.id = author_id

    class MockDiscordMessage:
        def __init__(self, content: str, author_id: str) -> None:
            self.content = content
            self.author = MockAuthor(author_id)

    raw_msg = MockDiscordMessage("test discord", "d456")
    response = await adapter.receive(raw_msg)

    assert response["handled_text"] == "test discord"
    assert response["handled_user"] == "d456"
    assert response["channel"] == "discord"

@pytest.mark.asyncio
async def test_telegram_adapter(gateway: GatewayRouter) -> None:
    """Test the TelegramAdapter receive logic.

    Args:
        gateway (GatewayRouter): The gateway fixture.
    """
    adapter = TelegramAdapter(gateway)

    class MockUser:
        def __init__(self, user_id: str) -> None:
            self.id = user_id

    class MockTelegramMessage:
        def __init__(self, text: str, user_id: str) -> None:
            self.text = text
            self.from_user = MockUser(user_id)

    raw_msg = MockTelegramMessage("test tg", "t789")
    response = await adapter.receive(raw_msg)

    assert response["handled_text"] == "test tg"
    assert response["handled_user"] == "t789"
    assert response["channel"] == "telegram"
