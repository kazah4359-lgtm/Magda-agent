import pytest
import asyncio
from typing import Any, Dict
from magda_agent.channels.base import ChannelAdapter
from magda_agent.channels.hub import ChannelHub
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage

class MockChannel(ChannelAdapter):
    """A mock channel adapter for testing purposes."""

    def __init__(self, channel_id: str, gateway: GatewayRouter) -> None:
        """Initialize the mock channel."""
        super().__init__(channel_id, gateway)

    async def receive(self, raw_data: Dict[str, Any]) -> Any:
        """Process incoming mock data and route it."""
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
        """Mock sending a message."""
        return f"Mock {self.channel_id} sent to {recipient_id}: {text}"

@pytest.fixture
def gateway() -> GatewayRouter:
    """Fixture that provides a GatewayRouter with a mock message handler."""
    router = GatewayRouter()

    async def mock_handler(msg: UnifiedMessage) -> Dict[str, str]:
        return {"handled_text": msg.text, "handled_user": msg.user_id, "channel": msg.channel}

    router.set_message_handler(mock_handler)
    return router

@pytest.fixture
def channel_hub() -> ChannelHub:
    """Fixture that provides a ChannelHub instance."""
    return ChannelHub()

def test_register_and_get_adapter(gateway: GatewayRouter, channel_hub: ChannelHub) -> None:
    """Test registering and retrieving adapters."""
    adapter1 = MockChannel("telegram_mock", gateway)
    adapter2 = MockChannel("discord_mock", gateway)

    channel_hub.register_adapter(adapter1)
    channel_hub.register_adapter(adapter2)

    assert channel_hub.get_adapter("telegram_mock") is adapter1
    assert channel_hub.get_adapter("discord_mock") is adapter2
    assert channel_hub.get_adapter("unknown") is None

@pytest.mark.asyncio
async def test_receive_from_channel(gateway: GatewayRouter, channel_hub: ChannelHub) -> None:
    """Test receiving messages from a specific channel through the hub."""
    adapter = MockChannel("telegram_mock", gateway)
    channel_hub.register_adapter(adapter)

    response = await channel_hub.receive_from_channel("telegram_mock", {"text": "hello hub", "user_id": "user1"})

    assert response is not None
    assert response["handled_text"] == "hello hub"
    assert response["handled_user"] == "user1"
    assert response["channel"] == "telegram_mock"

    # Test receiving from an unknown channel
    response_unknown = await channel_hub.receive_from_channel("unknown_channel", {"text": "hello"})
    assert response_unknown is None

@pytest.mark.asyncio
async def test_send_to_channel(gateway: GatewayRouter, channel_hub: ChannelHub) -> None:
    """Test dispatching outgoing messages to a specific channel through the hub."""
    adapter1 = MockChannel("telegram_mock", gateway)
    adapter2 = MockChannel("discord_mock", gateway)
    channel_hub.register_adapter(adapter1)
    channel_hub.register_adapter(adapter2)

    response1 = await channel_hub.send_to_channel("telegram_mock", "user1", "message 1")
    response2 = await channel_hub.send_to_channel("discord_mock", "user2", "message 2")

    assert response1 == "Mock telegram_mock sent to user1: message 1"
    assert response2 == "Mock discord_mock sent to user2: message 2"

    # Test sending to an unknown channel
    response_unknown = await channel_hub.send_to_channel("unknown_channel", "user3", "hello")
    assert response_unknown == "Error: Channel 'unknown_channel' not found."
