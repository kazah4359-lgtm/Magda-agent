import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from magda_agent.visualization.canvas_streaming_v2 import compute_diff, CanvasServerV2

def test_compute_diff_no_changes():
    old_state = json.dumps({"a": 1, "b": 2})
    new_state = json.dumps({"a": 1, "b": 2})
    assert compute_diff(old_state, new_state) is None

def test_compute_diff_changes():
    old_state = json.dumps({"a": 1, "b": 2})
    new_state = json.dumps({"a": 1, "b": 3, "c": 4})
    diff = compute_diff(old_state, new_state)
    assert diff == {"b": 3, "c": 4}

def test_compute_diff_deleted_keys():
    old_state = json.dumps({"a": 1, "b": 2})
    new_state = json.dumps({"a": 1})
    diff = compute_diff(old_state, new_state)
    assert diff == {"b": None}

def test_compute_diff_empty_old_state():
    old_state = ""
    new_state = json.dumps({"a": 1})
    diff = compute_diff(old_state, new_state)
    assert diff == {"a": 1}

@pytest.mark.asyncio
async def test_canvas_server_v2_connect():
    consciousness_mock = MagicMock()
    with patch('magda_agent.visualization.canvas_v3.CanvasVisualizerV3') as MockVisualizer:
        mock_vis_instance = MockVisualizer.return_value
        mock_vis_instance.get_state_json.return_value = json.dumps({"test": "initial"})

        server = CanvasServerV2(consciousness_mock)
        websocket_mock = AsyncMock()

        await server.connect(websocket_mock)

        websocket_mock.accept.assert_awaited_once()
        assert websocket_mock in server.active_connections

        # Check that it sent the initial full state
        expected_msg = json.dumps({"type": "full", "data": {"test": "initial"}})
        websocket_mock.send_text.assert_awaited_once_with(expected_msg)

@pytest.mark.asyncio
async def test_canvas_server_v2_disconnect():
    consciousness_mock = MagicMock()
    server = CanvasServerV2(consciousness_mock)
    websocket_mock = AsyncMock()

    # Fake connect
    server.active_connections.append(websocket_mock)
    assert len(server.active_connections) == 1

    server.disconnect(websocket_mock)
    assert len(server.active_connections) == 0

@pytest.mark.asyncio
async def test_canvas_server_v2_broadcast():
    consciousness_mock = MagicMock()
    server = CanvasServerV2(consciousness_mock)
    websocket_mock1 = AsyncMock()
    websocket_mock2 = AsyncMock()

    server.active_connections = [websocket_mock1, websocket_mock2]

    await server.broadcast("hello")

    websocket_mock1.send_text.assert_awaited_once_with("hello")
    websocket_mock2.send_text.assert_awaited_once_with("hello")

@pytest.mark.asyncio
async def test_canvas_server_v2_streaming_loop():
    consciousness_mock = MagicMock()

    with patch('magda_agent.visualization.canvas_v3.CanvasVisualizerV3') as MockVisualizer:
        mock_vis_instance = MockVisualizer.return_value

        # State transitions
        # 1. First iteration: "" -> {"a": 1} (Full)
        # 2. Second iteration: {"a": 1} -> {"a": 1, "b": 2} (Diff)
        # 3. Third iteration: {"a": 1, "b": 2} -> {"a": 1, "b": 2} (No change)

        states = [
            json.dumps({"a": 1}),
            json.dumps({"a": 1, "b": 2}),
            json.dumps({"a": 1, "b": 2}),
            json.dumps({"a": 1, "b": 2}),
        ]

        state_iter = iter(states)
        mock_vis_instance.get_state_json.side_effect = lambda: next(state_iter)

        server = CanvasServerV2(consciousness_mock, interval=0.01)
        websocket_mock = AsyncMock()
        server.active_connections.append(websocket_mock)

        # We need to run start_streaming in a task and let it run a few iterations, then stop it.
        # Alternatively, we mock asyncio.sleep to break the loop or advance state.

        loop_counter = 0
        original_sleep = asyncio.sleep

        async def mock_sleep(delay):
            nonlocal loop_counter
            loop_counter += 1
            if loop_counter >= 3:
                server._running = False
            await original_sleep(0) # yield control

        with patch('asyncio.sleep', side_effect=mock_sleep):
            await server.start_streaming()

        # Verify broadcasts
        assert websocket_mock.send_text.call_count == 2

        # First call should be type: full (because last_state was empty)
        call1 = websocket_mock.send_text.call_args_list[0][0][0]
        assert json.loads(call1) == {"type": "full", "data": {"a": 1}}

        # Second call should be type: diff
        call2 = websocket_mock.send_text.call_args_list[1][0][0]
        assert json.loads(call2) == {"type": "diff", "data": {"b": 2}}

        # Third iteration had no change, so no 3rd call
