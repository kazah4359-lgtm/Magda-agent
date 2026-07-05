import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage
from magda_agent.channels.multi_platform_v1 import SlackAdapter, MultiPlatformDispatcher

@pytest.fixture
def mock_gateway():
    gateway = MagicMock(spec=GatewayRouter)
    gateway.route_message = AsyncMock(return_value="Routed successfully")
    return gateway

@pytest.mark.asyncio
async def test_slack_adapter_receive_dict(mock_gateway):
    """Test receiving a dictionary via SlackAdapter."""
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
async def test_slack_adapter_receive_object(mock_gateway):
    """Test receiving an object via SlackAdapter."""
    class MockSlackMessage:
        def __init__(self, text, user):
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
async def test_slack_adapter_send(mock_gateway):
    """Test sending a message via SlackAdapter."""
    adapter = SlackAdapter(mock_gateway)

    result = await adapter.send("U123", "message to slack")
    assert result == "Slack sent to U123: message to slack"

@pytest.mark.asyncio
async def test_multi_platform_dispatcher_dispatch(mock_gateway):
    """Test MultiPlatformDispatcher delegates properly to different channels."""
    # Note: aiogram checks for a token in the format "ID:TOKEN", so we pass a mock valid one
    dispatcher = MultiPlatformDispatcher(mock_gateway, bot_token="12345:mock_token_fake")

    # Send via Slack
    result_slack = await dispatcher.dispatch("slack", "U123", "test slack")
    assert result_slack == "Slack sent to U123: test slack"

    # Send via Discord
    result_discord = await dispatcher.dispatch("discord", "111", "test discord")
    assert result_discord == "Discord sent to 111: test discord"

    # Send via Telegram (mocked no real bot but we'll check the output logic)
    # We pass a mock bot so it doesn't fail on network
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(return_value="Telegram mock sent")
    result_telegram = await dispatcher.dispatch("telegram", "222", "test tg", metadata={"bot": mock_bot})
    assert result_telegram == "Telegram mock sent"
    mock_bot.send_message.assert_called_once_with(chat_id="222", text="test tg")

@pytest.mark.asyncio
async def test_multi_platform_dispatcher_unknown_channel(mock_gateway):
    """Test MultiPlatformDispatcher with an unknown channel."""
    dispatcher = MultiPlatformDispatcher(mock_gateway)

    result = await dispatcher.dispatch("unknown", "333", "hello")
    assert result == "Error: Channel 'unknown' not found."
