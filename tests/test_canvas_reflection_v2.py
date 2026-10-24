import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.telemetry.canvas_reflection_v2 import CanvasReflectionTelemetryV2


def test_broadcast_reflection_event_send_text() -> None:
    """Test broadcasting a reflection V2 event when websocket has send_text."""
    async def _run() -> None:
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock()

        telemetry = CanvasReflectionTelemetryV2(websocket=mock_websocket)

        summary = "Completed multi-step V2 task successfully."
        lessons = ["Always verify file outputs."]
        anti_patterns = ["Do not skip pre-commit checks."]
        proposed_tasks = ["Improve telemetry coverage."]
        new_facts = ["Project uses pytest."]
        pad_adjustment = {"p": 0.1, "a": -0.05, "d": 0.2}
        reflection_id = "test-reflection-id-1234"

        with patch("time.time", return_value=1700000000.0):
            await telemetry.broadcast_reflection_event(
                summary=summary,
                lessons=lessons,
                anti_patterns=anti_patterns,
                proposed_tasks=proposed_tasks,
                new_facts=new_facts,
                pad_adjustment=pad_adjustment,
                reflection_id=reflection_id,
            )

        expected_payload = {
            "type": "subconsciousness_reflection_v2",
            "version": "v2",
            "timestamp": 1700000000.0,
            "data": {
                "reflection_id": reflection_id,
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


def test_broadcast_reflection_v2_alias() -> None:
    """Test broadcast_reflection_v2 alias method."""
    async def _run() -> None:
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock()

        telemetry = CanvasReflectionTelemetryV2(websocket=mock_websocket)

        summary = "V2 alias test summary."
        reflection_id = "test-alias-id-5678"

        with patch("time.time", return_value=1700000100.0):
            await telemetry.broadcast_reflection_v2(
                summary=summary,
                reflection_id=reflection_id,
            )

        expected_payload = {
            "type": "subconsciousness_reflection_v2",
            "version": "v2",
            "timestamp": 1700000100.0,
            "data": {
                "reflection_id": reflection_id,
                "summary": summary,
                "lessons": [],
                "anti_patterns": [],
                "proposed_tasks": [],
                "new_facts": [],
                "pad_adjustment": {"p": 0.0, "a": 0.0, "d": 0.0},
            },
        }
        expected_message = json.dumps(expected_payload)

        mock_websocket.send_text.assert_awaited_once_with(expected_message)

    asyncio.run(_run())


def test_broadcast_reflection_event_send_fallback() -> None:
    """Test broadcasting a reflection V2 event when websocket only has send."""
    async def _run() -> None:
        mock_websocket = AsyncMock()
        del mock_websocket.send_text
        mock_websocket.send = AsyncMock()

        telemetry = CanvasReflectionTelemetryV2(websocket=mock_websocket)

        summary = "Minimal reflection summary V2."

        with patch("time.time", return_value=1700000005.0), patch(
            "uuid.uuid4", return_value="generated-uuid-9999"
        ):
            await telemetry.broadcast_reflection_event(summary=summary)

        expected_payload = {
            "type": "subconsciousness_reflection_v2",
            "version": "v2",
            "timestamp": 1700000005.0,
            "data": {
                "reflection_id": "generated-uuid-9999",
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
        telemetry = CanvasReflectionTelemetryV2(websocket=None)
        telemetry.logger = MagicMock()

        await telemetry.broadcast_reflection_event("No websocket test")

        telemetry.logger.debug.assert_called_once_with(
            "Skipping broadcast for subconsciousness_reflection_v2 - no websocket connected."
        )

    asyncio.run(_run())


def test_broadcast_reflection_event_exception() -> None:
    """Test handling exceptions during broadcast gracefully."""
    async def _run() -> None:
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock(side_effect=Exception("Connection reset"))

        telemetry = CanvasReflectionTelemetryV2(websocket=mock_websocket)
        telemetry.logger = MagicMock()

        await telemetry.broadcast_reflection_event("Error test")

        telemetry.logger.error.assert_called_once()
        assert "Failed to broadcast subconsciousness_reflection_v2 event: Connection reset" in str(
            telemetry.logger.error.call_args
        )

    asyncio.run(_run())
