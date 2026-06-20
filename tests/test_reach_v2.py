import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.gateway.router import GatewayRouter
from magda_agent.channels.reach_v2 import SlackAdapter, WhatsAppAdapter, SignalAdapter, CLIAdapter

@pytest.mark.asyncio
async def test_slack_adapter() -> None:
    """Test SlackAdapter receive and send."""
    gateway = GatewayRouter()
    gateway.route_message = AsyncMock(return_value="routed")

    adapter = SlackAdapter(gateway)
    assert adapter.channel_id == "slack"

    # Test receive
    raw_data = {"text": "hello slack", "user": "user123"}
    res = await adapter.receive(raw_data)
    assert res == "routed"
    gateway.route_message.assert_called_once()
    msg = gateway.route_message.call_args[0][0]
    assert msg.channel == "slack"
    assert msg.text == "hello slack"
    assert msg.user_id == "user123"

    # Test send
    send_res = await adapter.send("user123", "response")
    assert send_res == "Slack sent to user123: response"

@pytest.mark.asyncio
async def test_whatsapp_adapter() -> None:
    """Test WhatsAppAdapter receive and send."""
    gateway = GatewayRouter()
    gateway.route_message = AsyncMock(return_value="routed")

    adapter = WhatsAppAdapter(gateway)
    assert adapter.channel_id == "whatsapp"

    # Test receive
    raw_data = {"body": "hello wa", "from": "wa_user"}
    res = await adapter.receive(raw_data)
    assert res == "routed"
    gateway.route_message.assert_called_once()
    msg = gateway.route_message.call_args[0][0]
    assert msg.channel == "whatsapp"
    assert msg.text == "hello wa"
    assert msg.user_id == "wa_user"

    # Test send
    send_res = await adapter.send("wa_user", "response")
    assert send_res == "WhatsApp sent to wa_user: response"

@pytest.mark.asyncio
async def test_signal_adapter() -> None:
    """Test SignalAdapter receive and send."""
    gateway = GatewayRouter()
    gateway.route_message = AsyncMock(return_value="routed")

    adapter = SignalAdapter(gateway)
    assert adapter.channel_id == "signal"

    # Test receive
    raw_data = {"message": "hello signal", "source": "sig_user"}
    res = await adapter.receive(raw_data)
    assert res == "routed"
    gateway.route_message.assert_called_once()
    msg = gateway.route_message.call_args[0][0]
    assert msg.channel == "signal"
    assert msg.text == "hello signal"
    assert msg.user_id == "sig_user"

    # Test send
    send_res = await adapter.send("sig_user", "response")
    assert send_res == "Signal sent to sig_user: response"

@pytest.mark.asyncio
async def test_cli_adapter() -> None:
    """Test CLIAdapter receive and send."""
    gateway = GatewayRouter()
    gateway.route_message = AsyncMock(return_value="routed")

    adapter = CLIAdapter(gateway)
    assert adapter.channel_id == "cli"

    # Test receive dictionary
    raw_data = {"input": "hello cli"}
    res = await adapter.receive(raw_data)
    assert res == "routed"
    gateway.route_message.assert_called_once()
    msg = gateway.route_message.call_args[0][0]
    assert msg.channel == "cli"
    assert msg.text == "hello cli"
    assert msg.user_id == "local_user"

    # Test send
    send_res = await adapter.send("local_user", "response")
    assert send_res == "CLI output for local_user: response"

    # Test receive string
    gateway.route_message.reset_mock()
    res_str = await adapter.receive("hello raw string")
    assert res_str == "routed"
    msg_str = gateway.route_message.call_args[0][0]
    assert msg_str.text == "hello raw string"
