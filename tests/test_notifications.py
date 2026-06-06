import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from magda_agent.notifications.manager import NotificationManager
from magda_agent.gateway.router import GatewayRouter

@pytest.mark.asyncio
async def test_send_notification_no_user_model():
    gateway = GatewayRouter()
    channel = AsyncMock()
    gateway.register_channel("telegram", channel)

    manager = NotificationManager(gateway=gateway)

    result = await manager.send_notification("123", "telegram", "Hello!")

    assert result is True
    channel.send.assert_awaited_once_with(recipient_id="123", text="Hello!", metadata={"notification_type": "general", "urgency": "normal"})

@pytest.mark.asyncio
async def test_send_notification_with_quiet_hours_blocked():
    gateway = GatewayRouter()
    channel = AsyncMock()
    gateway.register_channel("telegram", channel)

    user_model = MagicMock()
    user_model.get_model.return_value = {
        "preferences": {
            "quiet_hours": True
        }
    }

    manager = NotificationManager(gateway=gateway, user_model=user_model)

    result = await manager.send_notification("123", "telegram", "Hello!")

    assert result is False
    channel.send.assert_not_awaited()

@pytest.mark.asyncio
async def test_send_notification_with_quiet_hours_critical_urgency_allowed():
    gateway = GatewayRouter()
    channel = AsyncMock()
    gateway.register_channel("telegram", channel)

    user_model = MagicMock()
    user_model.get_model.return_value = {
        "preferences": {
            "quiet_hours": True
        }
    }

    manager = NotificationManager(gateway=gateway, user_model=user_model)

    result = await manager.send_notification("123", "telegram", "Urgent alert!", urgency="critical")

    assert result is True
    channel.send.assert_awaited_once()

@pytest.mark.asyncio
async def test_send_notification_ignored_type():
    gateway = GatewayRouter()
    channel = AsyncMock()
    gateway.register_channel("telegram", channel)

    user_model = MagicMock()
    user_model.get_model.return_value = {
        "preferences": {
            "ignored_notifications": ["report"]
        }
    }

    manager = NotificationManager(gateway=gateway, user_model=user_model)

    result = await manager.send_notification("123", "telegram", "Daily report", notification_type="report")

    assert result is False
    channel.send.assert_not_awaited()

@pytest.mark.asyncio
async def test_send_notification_invalid_channel():
    gateway = GatewayRouter()

    manager = NotificationManager(gateway=gateway)

    result = await manager.send_notification("123", "unknown_channel", "Hello!")

    assert result is False
