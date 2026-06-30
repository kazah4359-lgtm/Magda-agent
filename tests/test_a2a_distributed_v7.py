import pytest
from unittest.mock import AsyncMock, patch
from magda_agent.telemetry.a2a_distributed_v7 import A2ADistributedTelemetryV7

@pytest.fixture
def telemetry():
    return A2ADistributedTelemetryV7()

def test_track_event(telemetry):
    telemetry.track_event("sub_1", "task_start", {"task_id": 101})
    assert len(telemetry.events) == 1
    assert telemetry.events[0] == {
        "subagent_id": "sub_1",
        "event_name": "task_start",
        "payload": {"task_id": 101}
    }

@pytest.mark.asyncio
async def test_broadcast_events_empty(telemetry):
    with patch.object(telemetry, '_mock_broadcast', new_callable=AsyncMock) as mock_broadcast:
        await telemetry.broadcast_events()
        mock_broadcast.assert_not_called()

@pytest.mark.asyncio
async def test_broadcast_events(telemetry):
    telemetry.track_event("sub_1", "task_start", {"task_id": 101})
    telemetry.track_event("sub_2", "task_end", {"task_id": 102})

    assert len(telemetry.events) == 2

    with patch.object(telemetry, '_mock_broadcast', new_callable=AsyncMock) as mock_broadcast:
        await telemetry.broadcast_events()

        # Verify broadcast was called with correct payload
        mock_broadcast.assert_called_once()
        args, _ = mock_broadcast.call_args
        payload = args[0]

        assert payload["type"] == "telemetry_broadcast"
        assert len(payload["events"]) == 2
        assert payload["events"][0]["subagent_id"] == "sub_1"
        assert payload["events"][1]["subagent_id"] == "sub_2"

        # Verify events list is cleared
        assert len(telemetry.events) == 0
