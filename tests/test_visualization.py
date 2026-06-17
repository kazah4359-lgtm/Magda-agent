import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from fastapi import WebSocket, WebSocketDisconnect
from magda_agent.visualization.server import CanvasServer

@pytest.fixture
def mock_consciousness():
    mock = MagicMock()
    mock.emotions = None
    mock.mental_states = None
    mock.memory = None
    mock.skills = None
    mock.planner = None
    return mock

@pytest.fixture
def canvas_server(mock_consciousness):
    return CanvasServer(consciousness=mock_consciousness, interval=0.1)

@pytest.mark.asyncio
async def test_connect(canvas_server):
    mock_ws = AsyncMock(spec=WebSocket)
    await canvas_server.connect(mock_ws)
    assert mock_ws in canvas_server.active_connections
    mock_ws.accept.assert_awaited_once()

def test_disconnect(canvas_server):
    mock_ws = MagicMock(spec=WebSocket)
    canvas_server.active_connections.append(mock_ws)
    canvas_server.disconnect(mock_ws)
    assert mock_ws not in canvas_server.active_connections

@pytest.mark.asyncio
async def test_broadcast(canvas_server):
    mock_ws1 = AsyncMock(spec=WebSocket)
    mock_ws2 = AsyncMock(spec=WebSocket)

    canvas_server.active_connections.extend([mock_ws1, mock_ws2])

    await canvas_server.broadcast("Test Message")

    mock_ws1.send_text.assert_awaited_once_with("Test Message")
    mock_ws2.send_text.assert_awaited_once_with("Test Message")

@pytest.mark.asyncio
async def test_broadcast_disconnects_on_error(canvas_server):
    mock_ws1 = AsyncMock(spec=WebSocket)
    mock_ws2 = AsyncMock(spec=WebSocket)
    mock_ws2.send_text.side_effect = Exception("Connection closed")

    canvas_server.active_connections.extend([mock_ws1, mock_ws2])

    await canvas_server.broadcast("Test Message")

    mock_ws1.send_text.assert_awaited_once_with("Test Message")
    assert mock_ws1 in canvas_server.active_connections
    assert mock_ws2 not in canvas_server.active_connections

@pytest.mark.asyncio
async def test_start_streaming(canvas_server, mock_consciousness):
    mock_ws = AsyncMock(spec=WebSocket)
    canvas_server.active_connections.append(mock_ws)

    # Run start_streaming as a background task
    task = asyncio.create_task(canvas_server.start_streaming())

    # Yield control to the event loop so the task can run
    await asyncio.sleep(0.15)

    # Stop the stream
    await canvas_server.stop_streaming()
    await task

    # The interval is 0.1s, sleeping 0.15s should trigger at least 1 broadcast
    assert mock_ws.send_text.await_count >= 1
