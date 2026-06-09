import pytest
import asyncio
from magda_agent.gateway.canvas import CanvasVisualizer

@pytest.mark.asyncio
async def test_canvas_visualizer():
    canvas = CanvasVisualizer()
    q = canvas.subscribe()

    await canvas.update_state("active_task", "OpenClaw RL")

    assert canvas.get_state() == {"active_task": "OpenClaw RL"}

    msg = await asyncio.wait_for(q.get(), timeout=1.0)
    assert msg == {"type": "state_update", "key": "active_task", "value": "OpenClaw RL"}

    canvas.unsubscribe(q)
    await canvas.update_state("active_task", "Done")
    assert q.empty()
