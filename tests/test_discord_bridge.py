"""Tests for Discord Bridge module."""
import pytest
import asyncio
from unittest.mock import AsyncMock

from magda_agent.integration.discord_bridge import DiscordBridge

@pytest.mark.asyncio
async def test_discord_bridge_start_stop():
    """Test starting and stopping the Discord bridge."""
    mock_callback = AsyncMock()
    bridge = DiscordBridge("dummy", mock_callback)

    assert not bridge.is_running
    await bridge.start()
    assert bridge.is_running
    await bridge.stop()
    assert not bridge.is_running

@pytest.mark.asyncio
async def test_discord_bridge_on_message():
    """Test handling a message when the bridge is running."""
    mock_callback = AsyncMock(return_value="Hello from agent")
    bridge = DiscordBridge("dummy", mock_callback)

    await bridge.start()
    response = await bridge.on_message("123", "Hi there")

    assert response == "Hello from agent"
    mock_callback.assert_called_once_with("Hi there", 123)

@pytest.mark.asyncio
async def test_discord_bridge_stopped():
    """Test handling a message when the bridge is stopped."""
    mock_callback = AsyncMock()
    bridge = DiscordBridge("dummy", mock_callback)

    response = await bridge.on_message("123", "Hi there")

    assert response is None
    mock_callback.assert_not_called()

@pytest.mark.asyncio
async def test_discord_bridge_error_in_callback():
    """Test handling an error in the agent callback."""
    mock_callback = AsyncMock(side_effect=Exception("Test error"))
    bridge = DiscordBridge("dummy", mock_callback)

    await bridge.start()
    response = await bridge.on_message("123", "Hi there")

    assert response == "Error processing request."

@pytest.mark.asyncio
async def test_discord_bridge_send_message():
    """Test sending a message to Discord."""
    mock_callback = AsyncMock()
    bridge = DiscordBridge("dummy", mock_callback)

    await bridge.start()
    # Should not raise any exceptions
    await bridge.send_message("user123", "Response")

    await bridge.stop()
    await bridge.send_message("user123", "Response")
