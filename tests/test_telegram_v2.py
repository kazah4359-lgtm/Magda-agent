import pytest
from unittest.mock import AsyncMock, MagicMock
from aiogram.types import Message, User, Chat

from magda_agent.channels.telegram import TelegramAdapter
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage

@pytest.fixture
def gateway():
    router = GatewayRouter()

    async def mock_handler(msg: UnifiedMessage):
        return {"handled_text": msg.text, "handled_user": msg.user_id, "channel": msg.channel}

    router.set_message_handler(mock_handler)
    return router

@pytest.mark.asyncio
async def test_telegram_adapter_receive_message_obj(gateway):
    """Test receiving a real aiogram Message object."""
    adapter = TelegramAdapter(gateway)

    # Actually instantiate a Message object using kwargs
    # Message requires message_id, date, chat
    from datetime import datetime
    chat = Chat(id=1, type="private")
    user = User(id=12345, is_bot=False, first_name="Test")
    mock_msg = Message(message_id=1, date=datetime.now(), chat=chat, text="Hello aiogram", from_user=user)

    response = await adapter.receive(mock_msg)

    assert response["handled_text"] == "Hello aiogram"
    assert response["handled_user"] == "12345"
    assert response["channel"] == "telegram"

@pytest.mark.asyncio
async def test_telegram_adapter_receive_dict(gateway):
    """Test fallback receiving a dictionary."""
    adapter = TelegramAdapter(gateway)

    raw_dict = {"text": "Hello dict", "user_id": "67890"}
    response = await adapter.receive(raw_dict)

    assert response["handled_text"] == "Hello dict"
    assert response["handled_user"] == "67890"
    assert response["channel"] == "telegram"

@pytest.mark.asyncio
async def test_telegram_adapter_send_with_bot(gateway):
    """Test sending with a mock bot."""
    adapter = TelegramAdapter(gateway)

    mock_bot = AsyncMock()
    mock_bot.send_message.return_value = "Mocked Response"

    metadata = {"bot": mock_bot}

    result = await adapter.send("111", "Test Send", metadata=metadata)

    assert result == "Mocked Response"
    mock_bot.send_message.assert_called_once_with(chat_id="111", text="Test Send")

@pytest.mark.asyncio
async def test_telegram_adapter_send_no_bot(gateway):
    """Test sending without a bot."""
    adapter = TelegramAdapter(gateway)

    result = await adapter.send("222", "Test Fallback")

    assert result == "Telegram sent to 222: Test Fallback (Mock/No Token)"
