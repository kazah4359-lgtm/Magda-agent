"""
Tests for the Cross-Platform Unified Notification Broadcast Skill.
"""

import pytest
import asyncio
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock
from magda_agent.channels.hub import ChannelHub
from magda_agent.channels.base import ChannelAdapter
from magda_agent.gateway.router import GatewayRouter
from magda_agent.skills.notification_broadcast import (
    broadcast_notification_async,
    broadcast_notification,
)


class DummyAdapter(ChannelAdapter):
    """
    A dummy channel adapter class for testing notification broadcasts.
    """
    def __init__(self, channel_id: str, gateway: GatewayRouter) -> None:
        """
        Initialize the dummy channel adapter.

        Args:
            channel_id (str): The ID of the channel.
            gateway (GatewayRouter): The gateway router.
        """
        super().__init__(channel_id, gateway)
        self.send_mock = AsyncMock(return_value="sent")

    async def receive(self, raw_data: Any) -> Any:
        """
        Mock processing raw incoming data.

        Args:
            raw_data (Any): Raw incoming data.

        Returns:
            Any: Mocked output.
        """
        return "received"

    async def send(self, recipient_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """
        Mock sending a message.

        Args:
            recipient_id (str): The ID of the recipient.
            text (str): The message text.
            metadata (Optional[Dict[str, Any]]): Metadata.

        Returns:
            Any: Result of mock sending.
        """
        return await self.send_mock(recipient_id, text, metadata)


@pytest.mark.asyncio
async def test_broadcast_notification_no_message() -> None:
    """
    Verify that broadcasting an empty message returns an error status.
    """
    hub = ChannelHub()
    result = await broadcast_notification_async("", "user123", hub)
    assert result["status"] == "error"
    assert "cannot be empty" in result["message"]


@pytest.mark.asyncio
async def test_broadcast_notification_no_hub() -> None:
    """
    Verify that broadcasting without a ChannelHub returns an error status.
    """
    result = await broadcast_notification_async("Hello", "user123", None)
    assert result["status"] == "error"
    assert "ChannelHub is required" in result["message"]


@pytest.mark.asyncio
async def test_broadcast_notification_empty_hub() -> None:
    """
    Verify that broadcasting with an empty ChannelHub returns an error status.
    """
    hub = ChannelHub()
    result = await broadcast_notification_async("Hello", "user123", hub)
    assert result["status"] == "error"
    assert "No registered channels" in result["message"]


@pytest.mark.asyncio
async def test_broadcast_notification_async_success() -> None:
    """
    Verify that async broadcast successfully routes messages to all registered adapters.
    """
    gateway = GatewayRouter()
    hub = ChannelHub()
    adapter1 = DummyAdapter("telegram", gateway)
    adapter2 = DummyAdapter("discord", gateway)
    hub.register_adapter(adapter1)
    hub.register_adapter(adapter2)

    result = await broadcast_notification_async("Critical System Alert!", "default_user", hub)

    assert result["status"] == "success"
    assert "telegram" in result["results"]
    assert "discord" in result["results"]
    assert result["results"]["telegram"]["status"] == "success"
    assert result["results"]["discord"]["status"] == "success"

    adapter1.send_mock.assert_awaited_once_with("default_user", "Critical System Alert!", None)
    adapter2.send_mock.assert_awaited_once_with("default_user", "Critical System Alert!", None)


@pytest.mark.asyncio
async def test_broadcast_notification_async_partial_failure() -> None:
    """
    Verify that dynamic error isolation works when one channel fails but another succeeds.
    """
    gateway = GatewayRouter()
    hub = ChannelHub()
    adapter1 = DummyAdapter("telegram", gateway)
    adapter2 = DummyAdapter("discord", gateway)

    # Make telegram fail, discord succeed
    adapter1.send_mock.side_effect = RuntimeError("Telegram API is down")
    hub.register_adapter(adapter1)
    hub.register_adapter(adapter2)

    result = await broadcast_notification_async("System Alert!", "default_user", hub)

    # Since discord succeeded, overall status should be success
    assert result["status"] == "success"
    assert result["results"]["telegram"]["status"] == "error"
    assert "Telegram API is down" in result["results"]["telegram"]["message"]
    assert result["results"]["discord"]["status"] == "success"


@pytest.mark.asyncio
async def test_broadcast_notification_async_full_failure() -> None:
    """
    Verify that when all channels fail, the overall status is failed.
    """
    gateway = GatewayRouter()
    hub = ChannelHub()
    adapter1 = DummyAdapter("telegram", gateway)
    adapter2 = DummyAdapter("discord", gateway)

    adapter1.send_mock.side_effect = RuntimeError("Telegram down")
    adapter2.send_mock.side_effect = RuntimeError("Discord down")
    hub.register_adapter(adapter1)
    hub.register_adapter(adapter2)

    result = await broadcast_notification_async("System Alert!", "default_user", hub)

    assert result["status"] == "failed"
    assert result["results"]["telegram"]["status"] == "error"
    assert result["results"]["discord"]["status"] == "error"


@pytest.mark.asyncio
async def test_broadcast_notification_async_recipient_overrides() -> None:
    """
    Verify that recipient_id is overridden for specified channels when overrides are provided.
    """
    gateway = GatewayRouter()
    hub = ChannelHub()
    adapter1 = DummyAdapter("telegram", gateway)
    adapter2 = DummyAdapter("discord", gateway)
    hub.register_adapter(adapter1)
    hub.register_adapter(adapter2)

    overrides = {"telegram": "override_tg_user"}
    result = await broadcast_notification_async(
        "Critical Alert!", "default_user", hub, recipients=overrides
    )

    assert result["status"] == "success"
    adapter1.send_mock.assert_awaited_once_with("override_tg_user", "Critical Alert!", None)
    adapter2.send_mock.assert_awaited_once_with("default_user", "Critical Alert!", None)


def test_broadcast_notification_sync_success() -> None:
    """
    Verify that the synchronous wrapper runs successfully from a synchronous thread context.
    """
    # This test runs in a synchronous thread (no active running event loop)
    gateway = GatewayRouter()
    hub = ChannelHub()
    adapter1 = DummyAdapter("telegram", gateway)
    hub.register_adapter(adapter1)

    result = broadcast_notification("Sync Alert!", "sync_user", hub)

    assert result["status"] == "success"
    assert "telegram" in result["results"]
    assert result["results"]["telegram"]["status"] == "success"


@pytest.mark.asyncio
async def test_broadcast_notification_sync_error_if_loop_running() -> None:
    """
    Verify that the synchronous wrapper returns an error when called from an active running event loop.
    """
    gateway = GatewayRouter()
    hub = ChannelHub()
    adapter1 = DummyAdapter("telegram", gateway)
    hub.register_adapter(adapter1)

    # Calling synchronous wrapper inside an active async test (where an event loop is running)
    result = broadcast_notification("Alert!", "user", hub)

    assert "status" in result and result["status"] == "error"
    assert "called from an active event loop" in result["message"]
