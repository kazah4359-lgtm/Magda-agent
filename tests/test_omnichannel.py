import pytest
from magda_agent.skills.omnichannel import send_message
from magda_agent.gateway.router import GatewayRouter
from magda_agent.channels.telegram import TelegramAdapter
from magda_agent.channels.discord import DiscordAdapter

def test_send_message_telegram():
    result = send_message("telegram", "12345678", "Hello from test!")
    assert "Success" in result
    assert "Telegram" in result
    assert "12345678" in result

def test_send_message_whatsapp():
    result = send_message("whatsapp", "+1234567890", "Hello from test!")
    assert "Success" in result
    assert "WhatsApp" in result
    assert "+1234567890" in result

def test_send_message_email():
    result = send_message("email", "test@example.com", "Hello from test!")
    assert "Success" in result
    assert "Email" in result
    assert "test@example.com" in result

def test_send_message_unsupported():
    result = send_message("carrier_pigeon", "roof", "Hello?")
    assert "Error" in result
    assert "not supported" in result

def test_omnichannel_sync_with_gateway():
    gateway = GatewayRouter()
    tg = TelegramAdapter(gateway)
    dc = DiscordAdapter(gateway)

    res_tg = send_message("telegram", "user1", "Hello TG", gateway=gateway)
    assert res_tg == "Telegram sent to user1: Hello TG"

    res_dc = send_message("discord", "user2", "Hello DC", gateway=gateway)
    assert res_dc == "Discord sent to user2: Hello DC"

@pytest.mark.asyncio
async def test_send_message_async_awaits_gateway_errors():
    from magda_agent.skills.omnichannel import send_message_async

    class FailingChannel:
        async def send(self, recipient, message):
            raise RuntimeError("network down")

    gateway = GatewayRouter()
    gateway.register_channel("telegram", FailingChannel())

    result = await send_message_async("telegram", "user", "hello", gateway=gateway)
    assert result.startswith("Error: Failed to send telegram message")
    assert "network down" in result
