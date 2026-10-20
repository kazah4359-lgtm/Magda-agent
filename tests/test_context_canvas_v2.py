import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from magda_agent.memory.context_engine import ContextEngine
from magda_agent.telemetry.context_canvas_v2 import ContextCanvasVisualizerV2


class DummyPlugin:
    async def bootstrap(self, config):
        pass

    def before_retrieval(self, query, user_id):
        return query


class DummyMemoryEntry:
    def __init__(self, content, importance=0.8, user_id=1, tags=None):
        self.content = content
        self.importance = importance
        self.user_id = user_id
        self.tags = tags or ["test"]


def test_visualizer_initialization():
    visualizer = ContextCanvasVisualizerV2()
    assert visualizer.context_engine is None
    assert visualizer.websocket is None


def test_get_formatted_state_empty():
    visualizer = ContextCanvasVisualizerV2()
    state = visualizer.get_formatted_state()

    assert state["type"] == "context_engine_canvas_v2"
    assert "event_id" in state
    assert "timestamp" in state
    assert state["data"]["plugins"] == []
    assert state["data"]["registered_hooks"] == {}
    assert state["data"]["items"] == []
    assert state["data"]["item_count"] == 0
    assert state["data"]["stats"]["total_characters"] == 0


def test_get_formatted_state_with_engine_and_items():
    engine = ContextEngine()
    plugin = DummyPlugin()
    engine.register_plugin(plugin)

    visualizer = ContextCanvasVisualizerV2(context_engine=engine)

    entry1 = DummyMemoryEntry("Hello world", importance=0.9, tags=["greeting"])
    dict_entry = {"content": "Dictionary item", "importance": 0.5, "user_id": 2}

    state = visualizer.get_formatted_state(context_items=[entry1, dict_entry], extra_metadata={"session_id": "123"})

    assert len(state["data"]["plugins"]) == 1
    assert state["data"]["plugins"][0]["name"] == "DummyPlugin"
    assert state["data"]["registered_hooks"]["bootstrap"] == 1
    assert state["data"]["registered_hooks"]["before_retrieval"] == 1
    assert state["data"]["item_count"] == 2
    assert state["data"]["items"][0]["content"] == "Hello world"
    assert state["data"]["items"][0]["importance"] == 0.9
    assert state["data"]["items"][1]["content"] == "Dictionary item"
    assert state["data"]["metadata"]["session_id"] == "123"


def test_get_state_json():
    engine = ContextEngine()
    visualizer = ContextCanvasVisualizerV2(context_engine=engine)
    entry = DummyMemoryEntry("Sample text")

    json_str = visualizer.get_state_json(context_items=[entry])
    parsed = json.loads(json_str)

    assert parsed["type"] == "context_engine_canvas_v2"
    assert parsed["data"]["item_count"] == 1
    assert parsed["data"]["items"][0]["content"] == "Sample text"


@pytest.mark.asyncio
async def test_broadcast_state_websocket_send():
    class StandardWebSocket:
        async def send(self, message):
            pass

    mock_websocket = StandardWebSocket()
    mock_websocket.send = AsyncMock()

    visualizer = ContextCanvasVisualizerV2(websocket=mock_websocket)

    entry = DummyMemoryEntry("Broadcast item")
    await visualizer.broadcast_state(context_items=[entry])

    mock_websocket.send.assert_called_once()
    payload_str = mock_websocket.send.call_args[0][0]
    payload = json.loads(payload_str)
    assert payload["type"] == "context_engine_canvas_v2"
    assert payload["data"]["items"][0]["content"] == "Broadcast item"


@pytest.mark.asyncio
async def test_broadcast_state_send_text():
    mock_websocket = MagicMock()
    mock_websocket.send_text = AsyncMock()

    visualizer = ContextCanvasVisualizerV2(websocket=mock_websocket)
    entry = DummyMemoryEntry("Send text item")
    await visualizer.broadcast_state(context_items=[entry])

    mock_websocket.send_text.assert_called_once()
