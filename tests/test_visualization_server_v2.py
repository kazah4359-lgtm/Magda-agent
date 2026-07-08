import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import WebSocket
from magda_agent.visualization.server_v2 import CanvasServerV2 as CanvasServer
from typing import Any

@pytest.fixture
def mock_consciousness() -> MagicMock:
    """Provides a mocked Consciousness instance."""
    mock = MagicMock()
    mock.emotions = None
    mock.mental_states = None
    mock.memory = None
    mock.skills = None
    mock.planner = None
    mock.hypothalamus = None
    mock.global_workspace = None
    return mock

@pytest.fixture
def canvas_server(mock_consciousness: MagicMock) -> CanvasServer:
    """Provides a CanvasServerV2 instance initialized with a mocked visualizer."""
    with patch('magda_agent.visualization.canvas_v3.CanvasVisualizerV3') as mock_visualizer_class:
        mock_visualizer_instance = mock_visualizer_class.return_value
        mock_visualizer_instance.get_state_json.return_value = '{"test": "state"}'
        server = CanvasServer(consciousness=mock_consciousness, interval=0.1)
        # Verify the mock is being used
        assert server.visualizer == mock_visualizer_instance
        return server

@pytest.mark.asyncio
async def test_connect(canvas_server: CanvasServer) -> None:
    """Test that a new websocket connection is accepted and appended to active connections."""
    mock_ws = AsyncMock(spec=WebSocket)
    await canvas_server.connect(mock_ws)
    assert mock_ws in canvas_server.active_connections
    mock_ws.accept.assert_awaited_once()

def test_disconnect(canvas_server: CanvasServer) -> None:
    """Test that a websocket is removed from active connections upon disconnect."""
    mock_ws = MagicMock(spec=WebSocket)
    canvas_server.active_connections.append(mock_ws)
    canvas_server.disconnect(mock_ws)
    assert mock_ws not in canvas_server.active_connections

@pytest.mark.asyncio
async def test_broadcast(canvas_server: CanvasServer) -> None:
    """Test that broadcast sends the payload to all active connections."""
    mock_ws1 = AsyncMock(spec=WebSocket)
    mock_ws2 = AsyncMock(spec=WebSocket)

    canvas_server.active_connections.extend([mock_ws1, mock_ws2])

    await canvas_server.broadcast("Test Message")

    mock_ws1.send_text.assert_awaited_once_with("Test Message")
    mock_ws2.send_text.assert_awaited_once_with("Test Message")

@pytest.mark.asyncio
async def test_broadcast_disconnects_on_error(canvas_server: CanvasServer) -> None:
    """Test that connections failing to receive broadcast are removed."""
    mock_ws1 = AsyncMock(spec=WebSocket)
    mock_ws2 = AsyncMock(spec=WebSocket)
    mock_ws2.send_text.side_effect = Exception("Connection closed")

    canvas_server.active_connections.extend([mock_ws1, mock_ws2])

    await canvas_server.broadcast("Test Message")

    mock_ws1.send_text.assert_awaited_once_with("Test Message")
    assert mock_ws1 in canvas_server.active_connections
    assert mock_ws2 not in canvas_server.active_connections

@pytest.mark.asyncio
async def test_start_streaming(canvas_server: CanvasServer, mock_consciousness: MagicMock) -> None:
    """Test that start_streaming broadcasts at regular intervals."""
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
    mock_ws.send_text.assert_any_call('{"test": "state"}')
