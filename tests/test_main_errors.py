import pytest
from unittest.mock import MagicMock, patch
from aiogram.types import ErrorEvent, Update
from magda_agent.main import error_handler

@pytest.mark.asyncio
@patch('magda_agent.main.logging.error')
async def test_error_handler_logs_exception(mock_logging_error):
    # Create a mock exception and update
    mock_exception = Exception("Test exception")
    mock_update = MagicMock(spec=Update)

    # Create a mock ErrorEvent
    mock_event = MagicMock(spec=ErrorEvent)
    mock_event.update = mock_update
    mock_event.exception = mock_exception

    # Call the error handler
    await error_handler(mock_event)

    # Assert logging.error was called correctly
    mock_logging_error.assert_called_once_with(
        f"Update: {mock_update}\nException: {mock_exception}",
        exc_info=True
    )
