import pytest
import asyncio
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage
from magda_agent.channels.telegram import TelegramAdapter
from magda_agent.channels.discord import DiscordAdapter
from magda_agent.channels.rest import RestAdapter

class MockAgentCore:
    def __init__(self):
        self.received_messages = []

    async def handle_message(self, message: UnifiedMessage):
        self.received_messages.append(message)
        return f"Processed: {message.text}"

@pytest.mark.asyncio
async def test_telegram_channel_routing():
    gateway = GatewayRouter()
    agent = MockAgentCore()
    gateway.set_message_handler(agent.handle_message)

    channel = TelegramAdapter(gateway)

    raw_telegram_msg = {"text": "Hello Telegram", "user_id": "123"}
    response = await channel.receive(raw_telegram_msg)

    assert response == "Processed: Hello Telegram"
    assert len(agent.received_messages) == 1
    assert agent.received_messages[0].channel == "telegram"
    assert agent.received_messages[0].text == "Hello Telegram"
    assert agent.received_messages[0].user_id == "123"

@pytest.mark.asyncio
async def test_discord_channel_routing():
    gateway = GatewayRouter()
    agent = MockAgentCore()
    gateway.set_message_handler(agent.handle_message)

    channel = DiscordAdapter(gateway)

    raw_discord_msg = {"content": "Hello Discord", "author_id": "456"}
    response = await channel.receive(raw_discord_msg)

    assert response == "Processed: Hello Discord"
    assert len(agent.received_messages) == 1
    assert agent.received_messages[0].channel == "discord"
    assert agent.received_messages[0].text == "Hello Discord"
    assert agent.received_messages[0].user_id == "456"

@pytest.mark.asyncio
async def test_rest_channel_routing():
    gateway = GatewayRouter()
    agent = MockAgentCore()
    gateway.set_message_handler(agent.handle_message)

    channel = RestAdapter(gateway)

    raw_rest_msg = {"text": "Hello REST", "user_id": "789"}
    response = await channel.receive(raw_rest_msg)

    assert response == "Processed: Hello REST"
    assert len(agent.received_messages) == 1
    assert agent.received_messages[0].channel == "rest"
    assert agent.received_messages[0].text == "Hello REST"
    assert agent.received_messages[0].user_id == "789"

@pytest.mark.asyncio
async def test_channel_sending():
    gateway = GatewayRouter()
    tg = TelegramAdapter(gateway)
    dc = DiscordAdapter(gateway)
    rest = RestAdapter(gateway)

    assert "Telegram sent to 1: hi (Mock/No Token)" == await tg.send("1", "hi")
    assert "Discord sent to 2: hello" == await dc.send("2", "hello")

    rest_resp = await rest.send("3", "hey")
    assert rest_resp["status"] == "success"
    assert rest_resp["recipient"] == "3"
    assert rest_resp["text"] == "hey"

@pytest.mark.asyncio
async def test_missing_handler():
    gateway = GatewayRouter()
    channel = TelegramAdapter(gateway)
    raw_telegram_msg = {"text": "Fail", "user_id": "1"}

    with pytest.raises(RuntimeError, match="No message handler registered with GatewayRouter"):
        await channel.receive(raw_telegram_msg)

from magda_agent.architecture.gateway import LocalFirstGateway

@pytest.mark.asyncio
async def test_local_first_gateway_routing_safely():
    gateway = LocalFirstGateway()
    agent = MockAgentCore()
    gateway.set_message_handler(agent.handle_message)

    msg = UnifiedMessage(channel="local", text="Hello Control Plane", user_id="123")
    response = await gateway.route_message(msg)

    assert response == "Processed: Hello Control Plane"
    assert len(agent.received_messages) == 1
    assert agent.received_messages[0].channel == "local"
    assert agent.received_messages[0].text == "Hello Control Plane"

@pytest.mark.asyncio
async def test_local_first_gateway_missing_handler():
    gateway = LocalFirstGateway()
    msg = UnifiedMessage(channel="local", text="Fail", user_id="123")

    with pytest.raises(RuntimeError, match="No message handler registered with LocalFirstGateway"):
        await gateway.route_message(msg)

def test_local_first_gateway_channel_registration():
    gateway = LocalFirstGateway()
    gateway.register_channel("telegram", "TelegramAdapter")
    assert gateway.get_channel("telegram") == "TelegramAdapter"
    assert gateway.get_channel("discord") is None
