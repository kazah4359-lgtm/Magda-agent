import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage
from magda_agent.channels.slack import SlackAdapter

@pytest.fixture
def mock_gateway() -> GatewayRouter:
    """Fixture to provide a mocked GatewayRouter.

    Returns:
        GatewayRouter: The mocked gateway.
    """
    gateway = MagicMock(spec=GatewayRouter)
    gateway.route_message = AsyncMock(return_value="Routed successfully")
    return gateway

@pytest.mark.asyncio
async def test_slack_adapter_receive_dict(mock_gateway: GatewayRouter) -> None:
    """Test receiving a dictionary via SlackAdapter.

    Args:
        mock_gateway (GatewayRouter): The mocked gateway router.
    """
    adapter = SlackAdapter(mock_gateway)
    raw_data = {"text": "hello slack", "user": "U12345"}

    result = await adapter.receive(raw_data)

    assert result == "Routed successfully"
    mock_gateway.route_message.assert_called_once()
    msg = mock_gateway.route_message.call_args[0][0]
    assert isinstance(msg, UnifiedMessage)
    assert msg.channel == "slack"
    assert msg.text == "hello slack"
    assert msg.user_id == "U12345"

@pytest.mark.asyncio
async def test_slack_adapter_receive_object(mock_gateway: GatewayRouter) -> None:
    """Test receiving an object via SlackAdapter.

    Args:
        mock_gateway (GatewayRouter): The mocked gateway router.
    """
    class MockSlackMessage:
        def __init__(self, text: str, user: str) -> None:
            self.text = text
            self.user = user

    adapter = SlackAdapter(mock_gateway)
    raw_data = MockSlackMessage("hey object", "U999")

    result = await adapter.receive(raw_data)

    assert result == "Routed successfully"
    mock_gateway.route_message.assert_called_once()
    msg = mock_gateway.route_message.call_args[0][0]
    assert msg.text == "hey object"
    assert msg.user_id == "U999"

@pytest.mark.asyncio
async def test_slack_adapter_send(mock_gateway: GatewayRouter) -> None:
    """Test sending a message via SlackAdapter.

    Args:
        mock_gateway (GatewayRouter): The mocked gateway router.
    """
    adapter = SlackAdapter(mock_gateway)

    result = await adapter.send("U123", "message to slack")
    assert result == "Slack sent to U123: message to slack"
