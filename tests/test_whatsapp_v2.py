import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage
from magda_agent.channels.whatsapp_v2 import WhatsAppAdapterV2

@pytest.fixture
def gateway() -> GatewayRouter:
    """Fixture to provide a GatewayRouter with mocked message routing behavior.

    Returns:
        GatewayRouter: A gateway router instance with AsyncMock routing.
    """
    router = GatewayRouter()
    router.route_message = AsyncMock(return_value="whatsapp_routed")
    return router

@pytest.mark.asyncio
async def test_whatsapp_v2_receive_dict(gateway: GatewayRouter) -> None:
    """Test receiving a WhatsApp message with raw data as a dictionary.

    Args:
        gateway (GatewayRouter): The gateway router fixture.
    """
    adapter = WhatsAppAdapterV2(gateway)
    assert adapter.channel_id == "whatsapp_v2"

    raw_data: Dict[str, Any] = {
        "body": "Hello WhatsApp V2",
        "from": "whatsapp_user_1"
    }

    res: Any = await adapter.receive(raw_data)
    assert res == "whatsapp_routed"

    gateway.route_message.assert_called_once()
    msg: UnifiedMessage = gateway.route_message.call_args[0][0]
    assert msg.channel == "whatsapp_v2"
    assert msg.text == "Hello WhatsApp V2"
    assert msg.user_id == "whatsapp_user_1"
    assert msg.metadata["raw"] == raw_data

@pytest.mark.asyncio
async def test_whatsapp_v2_receive_object(gateway: GatewayRouter) -> None:
    """Test receiving a WhatsApp message with raw data as a generic object.

    Args:
        gateway (GatewayRouter): The gateway router fixture.
    """
    adapter = WhatsAppAdapterV2(gateway)

    class RawObjectMessage:
        """A simple helper mock object for WhatsApp message testing."""

        def __init__(self, text: str, sender_id: str) -> None:
            """Initialize raw object message properties.

            Args:
                text (str): Message content.
                sender_id (str): Message sender ID.
            """
            self.text = text
            self.sender_id = sender_id

    raw_data = RawObjectMessage("Hi there", "wa_sender_99")

    res: Any = await adapter.receive(raw_data)
    assert res == "whatsapp_routed"

    gateway.route_message.assert_called_once()
    msg: UnifiedMessage = gateway.route_message.call_args[0][0]
    assert msg.channel == "whatsapp_v2"
    assert msg.text == "Hi there"
    assert msg.user_id == "wa_sender_99"
    assert msg.metadata["raw"] == raw_data

@pytest.mark.asyncio
async def test_whatsapp_v2_send(gateway: GatewayRouter) -> None:
    """Test sending an outbound message using the WhatsApp V2 adapter.

    Args:
        gateway (GatewayRouter): The gateway router fixture.
    """
    adapter = WhatsAppAdapterV2(gateway)
    res: Any = await adapter.send("wa_recipient", "Automated reply")
    assert res == "WhatsApp V2 sent to wa_recipient: Automated reply"
