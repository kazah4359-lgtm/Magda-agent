import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import WebSocket, WebSocketDisconnect

from magda_agent.visualization.server import CanvasServer
from magda_agent.visualization.canvas_api_v2 import get_canvas_v2_router

@pytest.mark.asyncio
async def test_canvas_stream_success() -> None:
    """Tests successful canvas websocket stream connection and message parsing."""
    mock_canvas_server = MagicMock(spec=CanvasServer)
    mock_canvas_server.connect = AsyncMock()
    mock_canvas_server.disconnect = MagicMock()

    router = get_canvas_v2_router(mock_canvas_server, token="secret")

    # Extract the route function directly
    stream_func = router.routes[0].endpoint

    mock_ws = AsyncMock(spec=WebSocket)
    # Simulate receiving text once, then disconnecting
    mock_ws.receive_text.side_effect = [ "ping", WebSocketDisconnect() ]

    await stream_func(mock_ws, auth_token="secret")

    mock_canvas_server.connect.assert_awaited_once_with(mock_ws)
    mock_canvas_server.disconnect.assert_called_once_with(mock_ws)

@pytest.mark.asyncio
async def test_canvas_stream_unauthorized() -> None:
    """Tests unauthorized connection closing with code 1008."""
    mock_canvas_server = MagicMock(spec=CanvasServer)
    mock_canvas_server.connect = AsyncMock()

    router = get_canvas_v2_router(mock_canvas_server, token="secret")
    stream_func = router.routes[0].endpoint

    mock_ws = AsyncMock(spec=WebSocket)

    await stream_func(mock_ws, auth_token="wrong")

    mock_ws.close.assert_awaited_once_with(code=1008)
    mock_canvas_server.connect.assert_not_called()
