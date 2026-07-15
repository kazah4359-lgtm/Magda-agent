import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from fastapi import WebSocket
from magda_agent.visualization.live_canvas import LiveCanvasStreamer

@pytest.fixture
def mock_consciousness() -> MagicMock:
    """
    Creates a mocked Consciousness instance to avoid real database/external dependencies.
    """
    mock = MagicMock()
    mock.emotions = None
    mock.mental_states = None
    mock.memory = None
    mock.skills = None
    mock.planner = None
    mock.hypothalamus = None
    mock.global_workspace = None
    mock.openclaw_rl_metrics = None
    return mock

@pytest.fixture
def live_canvas_streamer(mock_consciousness: MagicMock) -> LiveCanvasStreamer:
    """
    Provides a LiveCanvasStreamer instance with 0.1s update interval.
    """
    return LiveCanvasStreamer(consciousness=mock_consciousness, interval=0.1)

@pytest.mark.asyncio
async def test_live_canvas_connect(live_canvas_streamer: LiveCanvasStreamer) -> None:
    """
    Tests that a client connection is accepted and initial full state is immediately sent.
    """
    mock_ws = AsyncMock(spec=WebSocket)
    await live_canvas_streamer.connect(mock_ws)

    # Check connection added and accepted
    assert mock_ws in live_canvas_streamer.active_connections
    mock_ws.accept.assert_awaited_once()

    # Verify initial state sent
    mock_ws.send_text.assert_awaited_once()
    sent_text = mock_ws.send_text.await_args[0][0]
    sent_data = json.loads(sent_text)
    assert "emotions" in sent_data
    assert "memory" in sent_data

@pytest.mark.asyncio
async def test_live_canvas_disconnect(live_canvas_streamer: LiveCanvasStreamer) -> None:
    """
    Tests that a client connection is removed from active pool on disconnect.
    """
    mock_ws = MagicMock(spec=WebSocket)
    live_canvas_streamer.active_connections.append(mock_ws)
    live_canvas_streamer.disconnect(mock_ws)
    assert mock_ws not in live_canvas_streamer.active_connections

@pytest.mark.asyncio
async def test_live_canvas_broadcast(live_canvas_streamer: LiveCanvasStreamer) -> None:
    """
    Tests that state update broadcasts are correctly sent to all active connections.
    """
    mock_ws1 = AsyncMock(spec=WebSocket)
    mock_ws2 = AsyncMock(spec=WebSocket)

    live_canvas_streamer.active_connections.extend([mock_ws1, mock_ws2])
    test_msg = json.dumps({"test": "data"})
    await live_canvas_streamer.broadcast(test_msg)

    mock_ws1.send_text.assert_awaited_once_with(test_msg)
    mock_ws2.send_text.assert_awaited_once_with(test_msg)

@pytest.mark.asyncio
async def test_live_canvas_broadcast_failures(live_canvas_streamer: LiveCanvasStreamer) -> None:
    """
    Tests that a failing WebSocket connection during broadcast is automatically disconnected.
    """
    mock_ws1 = AsyncMock(spec=WebSocket)
    mock_ws2 = AsyncMock(spec=WebSocket)
    mock_ws2.send_text.side_effect = Exception("WebSocket closed by remote host")

    live_canvas_streamer.active_connections.extend([mock_ws1, mock_ws2])
    test_msg = json.dumps({"test": "data"})
    await live_canvas_streamer.broadcast(test_msg)

    mock_ws1.send_text.assert_awaited_once_with(test_msg)
    assert mock_ws1 in live_canvas_streamer.active_connections
    assert mock_ws2 not in live_canvas_streamer.active_connections

@pytest.mark.asyncio
async def test_live_canvas_streaming_loop(live_canvas_streamer: LiveCanvasStreamer) -> None:
    """
    Tests that the background streaming loop periodically sends state updates to clients.
    """
    mock_ws = AsyncMock(spec=WebSocket)
    live_canvas_streamer.active_connections.append(mock_ws)

    # Start the background task
    task = asyncio.create_task(live_canvas_streamer.start_streaming())

    # Yield control to the event loop
    await asyncio.sleep(0.15)

    # Stop streaming and await task completion
    await live_canvas_streamer.stop_streaming()
    await task

    # Assert that a broadcast occurred
    assert mock_ws.send_text.await_count >= 1
