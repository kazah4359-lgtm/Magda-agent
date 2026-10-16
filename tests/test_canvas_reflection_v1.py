import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.telemetry.canvas_reflection_v1 import CanvasReflectionTelemetryV1


def test_broadcast_reflection_event_send_text() -> None:
    """Test broadcasting a reflection event when websocket has send_text."""
    async def _run() -> None:
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock()

        telemetry = CanvasReflectionTelemetryV1(websocket=mock_websocket)

        summary = "Completed multi-step task successfully."
        lessons = ["Always verify file outputs."]
        anti_patterns = ["Do not skip pre-commit checks."]
        proposed_tasks = ["Improve telemetry coverage."]
        new_facts = ["Project uses pytest."]
        pad_adjustment = {"p": 0.1, "a": -0.05, "d": 0.2}

        with patch("time.time", return_value=1700000000.0):
            await telemetry.broadcast_reflection_event(
                summary=summary,
                lessons=lessons,
                anti_patterns=anti_patterns,
                proposed_tasks=proposed_tasks,
                new_facts=new_facts,
                pad_adjustment=pad_adjustment,
            )

        expected_payload = {
            "type": "subconsciousness_reflection",
            "timestamp": 1700000000.0,
            "data": {
                "summary": summary,
                "lessons": lessons,
                "anti_patterns": anti_patterns,
                "proposed_tasks": proposed_tasks,
                "new_facts": new_facts,
                "pad_adjustment": pad_adjustment,
            },
        }
        expected_message = json.dumps(expected_payload)

        mock_websocket.send_text.assert_awaited_once_with(expected_message)

    asyncio.run(_run())


def test_broadcast_reflection_event_send_fallback() -> None:
    """Test broadcasting a reflection event when websocket only has send."""
    async def _run() -> None:
        mock_websocket = AsyncMock()
        del mock_websocket.send_text
        mock_websocket.send = AsyncMock()

        telemetry = CanvasReflectionTelemetryV1(websocket=mock_websocket)

        summary = "Minimal reflection summary."

        with patch("time.time", return_value=1700000005.0):
            await telemetry.broadcast_reflection_event(summary=summary)

        expected_payload = {
            "type": "subconsciousness_reflection",
            "timestamp": 1700000005.0,
            "data": {
                "summary": summary,
                "lessons": [],
                "anti_patterns": [],
                "proposed_tasks": [],
                "new_facts": [],
                "pad_adjustment": {"p": 0.0, "a": 0.0, "d": 0.0},
            },
        }
        expected_message = json.dumps(expected_payload)

        mock_websocket.send.assert_awaited_once_with(expected_message)

    asyncio.run(_run())


def test_broadcast_reflection_event_no_websocket() -> None:
    """Test broadcasting when no websocket is provided."""
    async def _run() -> None:
        telemetry = CanvasReflectionTelemetryV1(websocket=None)
        telemetry.logger = MagicMock()

        await telemetry.broadcast_reflection_event("No websocket test")

        telemetry.logger.debug.assert_called_once_with(
            "Skipping broadcast for subconsciousness_reflection - no websocket connected."
        )

    asyncio.run(_run())


def test_broadcast_reflection_event_exception() -> None:
    """Test handling exceptions during broadcast gracefully."""
    async def _run() -> None:
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock(side_effect=Exception("Connection reset"))

        telemetry = CanvasReflectionTelemetryV1(websocket=mock_websocket)
        telemetry.logger = MagicMock()

        await telemetry.broadcast_reflection_event("Error test")

        telemetry.logger.error.assert_called_once()
        assert "Failed to broadcast subconsciousness_reflection event: Connection reset" in str(
            telemetry.logger.error.call_args
        )

    asyncio.run(_run())
