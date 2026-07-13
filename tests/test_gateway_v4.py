import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.architecture.gateway_v4 import LocalFirstGatewayV4, UnifiedRoutingLayer
from magda_agent.gateway.router import UnifiedMessage

@pytest.mark.asyncio
async def test_gateway_routing_basic():
    """Test basic routing from gateway to handler."""
    mock_handler = AsyncMock(return_value="processed")
    gateway = LocalFirstGatewayV4(mock_handler)

    msg = UnifiedMessage(channel="telegram", text="hello", user_id="user1")
    result = await gateway.route(msg)

    assert result == "processed"
    mock_handler.assert_called_once_with(msg)

@pytest.mark.asyncio
async def test_gateway_middleware_chain():
    """Test that middleware chain executes in order."""
    execution_order = []

    async def middleware1(msg, next_call):
        execution_order.append("m1_pre")
        res = await next_call(msg)
        execution_order.append("m1_post")
        return res

    async def middleware2(msg, next_call):
        execution_order.append("m2_pre")
        res = await next_call(msg)
        execution_order.append("m2_post")
        return res

    async def final_handler(msg):
        execution_order.append("handler")
        return "ok"

    gateway = LocalFirstGatewayV4(final_handler)
    gateway.use(middleware1)
    gateway.use(middleware2)

    msg = UnifiedMessage(channel="discord", text="test", user_id="user2")
    result = await gateway.route(msg)

    assert result == "ok"
    assert execution_order == ["m1_pre", "m2_pre", "handler", "m2_post", "m1_post"]

@pytest.mark.asyncio
async def test_gateway_channel_registration():
    """Test channel registration and retrieval."""
    gateway = LocalFirstGatewayV4(AsyncMock())
    mock_channel = MagicMock()

    gateway.register_channel("slack", mock_channel)
    assert gateway.get_channel("slack") == mock_channel
    assert gateway.get_channel("unknown") is None

@pytest.mark.asyncio
async def test_middleware_can_modify_message():
    """Test that middleware can modify the message before it reaches the handler."""
    async def modify_middleware(msg, next_call):
        msg.text = msg.text.upper()
        return await next_call(msg)

    async def handler(msg):
        return msg.text

    gateway = LocalFirstGatewayV4(handler)
    gateway.use(modify_middleware)

    msg = UnifiedMessage(channel="cli", text="shout", user_id="user3")
    result = await gateway.route(msg)

    assert result == "SHOUT"
