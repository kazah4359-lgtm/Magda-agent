import pytest
from magda_agent.channels.signal_v2 import SignalAdapterV2
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage
from typing import Dict, Any

@pytest.fixture
def gateway() -> GatewayRouter:
    """Fixture that provides a GatewayRouter with a mock message handler."""
    router = GatewayRouter()

    async def mock_handler(msg: UnifiedMessage) -> Dict[str, str]:
        """Mock handler for testing routed messages."""
        return {
            "handled_text": msg.text,
            "handled_user": msg.user_id,
            "channel": msg.channel
        }

    router.set_message_handler(mock_handler)
    return router

@pytest.mark.asyncio
async def test_signal_adapter_v2_receive_dict(gateway: GatewayRouter) -> None:
    """Test SignalAdapterV2 receiving a dictionary payload."""
    adapter = SignalAdapterV2(gateway)

    # Verify registration
    assert gateway.get_channel("signal_v2") is adapter

    raw_data = {"body": "hello signal", "source": "+123456789"}

    response = await adapter.receive(raw_data)

    assert response["handled_text"] == "hello signal"
    assert response["handled_user"] == "+123456789"
    assert response["channel"] == "signal_v2"

@pytest.mark.asyncio
async def test_signal_adapter_v2_receive_object(gateway: GatewayRouter) -> None:
    """Test SignalAdapterV2 receiving an object payload."""
    adapter = SignalAdapterV2(gateway)

    class MockSignalMessage:
        def __init__(self, body: str, source: str):
            self.body = body
            self.source = source

    raw_msg = MockSignalMessage("signal object", "+987654321")
    response = await adapter.receive(raw_msg)

    assert response["handled_text"] == "signal object"
    assert response["handled_user"] == "+987654321"
    assert response["channel"] == "signal_v2"

@pytest.mark.asyncio
async def test_signal_adapter_v2_send(gateway: GatewayRouter) -> None:
    """Test SignalAdapterV2 sending logic."""
    adapter = SignalAdapterV2(gateway)
    sent_response = await adapter.send("+123456789", "test message")

    assert "Signal V2 sent to +123456789: test message" in sent_response
