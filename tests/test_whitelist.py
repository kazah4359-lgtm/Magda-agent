import os
import pytest
from unittest.mock import AsyncMock, MagicMock
from aiogram.types import Message, User
from magda_agent.main import WhitelistMiddleware

@pytest.fixture
def mock_handler():
    return AsyncMock(return_value="handled")

@pytest.fixture
def create_message():
    def _create_message(user_id):
        message = MagicMock(spec=Message)
        user = MagicMock(spec=User)
        user.id = user_id
        message.from_user = user
        return message
    return _create_message

@pytest.mark.asyncio
async def test_whitelist_allows_authorized_user(monkeypatch, mock_handler, create_message):
    monkeypatch.setenv("ALLOWED_USER_IDS", "123,456")
    middleware = WhitelistMiddleware()
    message = create_message(123)

    result = await middleware(mock_handler, message, {})

    assert result == "handled"
    mock_handler.assert_called_once_with(message, {})

@pytest.mark.asyncio
async def test_whitelist_blocks_unauthorized_user(monkeypatch, mock_handler, create_message):
    monkeypatch.setenv("ALLOWED_USER_IDS", "123,456")
    middleware = WhitelistMiddleware()
    message = create_message(789)

    result = await middleware(mock_handler, message, {})

    assert result is None
    mock_handler.assert_not_called()

@pytest.mark.asyncio
async def test_whitelist_allows_all_if_not_configured(monkeypatch, mock_handler, create_message):
    monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
    middleware = WhitelistMiddleware()
    message = create_message(789)

    result = await middleware(mock_handler, message, {})

    assert result == "handled"
    mock_handler.assert_called_once_with(message, {})

@pytest.mark.asyncio
async def test_whitelist_handles_invalid_config(monkeypatch, mock_handler, create_message):
    monkeypatch.setenv("ALLOWED_USER_IDS", "invalid_id")
    middleware = WhitelistMiddleware()
    message = create_message(123)

    result = await middleware(mock_handler, message, {})

    # If invalid, it defaults to empty whitelist (allows all)
    assert result == "handled"
    mock_handler.assert_called_once_with(message, {})
