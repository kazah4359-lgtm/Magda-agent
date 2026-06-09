import pytest
import asyncio
from typing import Any
from unittest.mock import AsyncMock
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage
from magda_agent.gateway.channels import MockChannel, TelegramChannel, DiscordChannel

@pytest.mark.asyncio
async def test_mock_channel():
    router = GatewayRouter()
    handler = AsyncMock()
    router.set_message_handler(handler)
    channel = MockChannel("mock1")
    router.register_channel("mock1", channel)

    raw = {"text": "hello", "user_id": "u1"}
    await channel.process(raw, router)

    handler.assert_called_once()
    msg: UnifiedMessage = handler.call_args[0][0]
    assert msg.channel == "mock1"
    assert msg.text == "hello"
    assert msg.user_id == "u1"

@pytest.mark.asyncio
async def test_telegram_channel():
    router = GatewayRouter()
    handler = AsyncMock()
    router.set_message_handler(handler)
    channel = TelegramChannel()
    router.register_channel("telegram", channel)

    raw = {"message": {"text": "tg hello", "from": {"id": 123}}}
    await channel.process(raw, router)

    handler.assert_called_once()
    msg: UnifiedMessage = handler.call_args[0][0]
    assert msg.channel == "telegram"
    assert msg.text == "tg hello"
    assert msg.user_id == "123"

@pytest.mark.asyncio
async def test_discord_channel():
    router = GatewayRouter()
    handler = AsyncMock()
    router.set_message_handler(handler)
    channel = DiscordChannel()
    router.register_channel("discord", channel)

    raw = {"content": "dc hello", "author": {"id": "456"}}
    await channel.process(raw, router)

    handler.assert_called_once()
    msg: UnifiedMessage = handler.call_args[0][0]
    assert msg.channel == "discord"
    assert msg.text == "dc hello"
    assert msg.user_id == "456"
