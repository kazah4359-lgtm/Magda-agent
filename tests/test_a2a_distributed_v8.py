"""Tests for A2A Distributed Telemetry V8."""

import pytest
from unittest.mock import AsyncMock, patch

from magda_agent.telemetry.a2a_distributed_v8 import A2ADistributedTelemetryV8


@pytest.fixture
def telemetry() -> A2ADistributedTelemetryV8:
    """Fixture providing a telemetry instance."""
    return A2ADistributedTelemetryV8()


def test_track_event(telemetry: A2ADistributedTelemetryV8) -> None:
    """Test tracking a single event."""
    telemetry.track_event("sub_1", "task_start", {"task_id": 101})
    assert len(telemetry.events) == 1
    assert telemetry.events[0] == {
        "subagent_id": "sub_1",
        "event_name": "task_start",
        "payload": {"task_id": 101}
    }


def test_broadcast_events_empty(telemetry: A2ADistributedTelemetryV8) -> None:
    """Test broadcasting when there are no events."""
    import asyncio
    with patch.object(telemetry, '_mock_broadcast', new_callable=AsyncMock) as mock_broadcast:
        asyncio.run(telemetry.broadcast_events())
        mock_broadcast.assert_not_called()


def test_broadcast_events(telemetry: A2ADistributedTelemetryV8) -> None:
    """Test broadcasting collected events."""
    import asyncio
    telemetry.track_event("sub_1", "task_start", {"task_id": 101})
    telemetry.track_event("sub_2", "task_end", {"task_id": 102})

    assert len(telemetry.events) == 2

    with patch.object(telemetry, '_mock_broadcast', new_callable=AsyncMock) as mock_broadcast:
        asyncio.run(telemetry.broadcast_events())

        mock_broadcast.assert_called_once()
        args, _ = mock_broadcast.call_args
        payload = args[0]

        assert payload["type"] == "telemetry_broadcast"
        assert payload["version"] == "v8"
        assert len(payload["events"]) == 2
        assert payload["events"][0]["subagent_id"] == "sub_1"
        assert payload["events"][1]["subagent_id"] == "sub_2"

        # Check queue is cleared
        assert len(telemetry.events) == 0
