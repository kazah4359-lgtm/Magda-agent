import pytest
import json
from typing import Any
from unittest.mock import MagicMock
from magda_agent.memory.context_engine_visualizer import ContextEngineVisualizer
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.memory.context_engine_v5 import ContextEngineV5
from magda_agent.memory.working import MemoryEntry

class DummyEmotionalState:
    """Dummy class to represent an emotional state with pleasure, arousal, dominance."""
    def __init__(self, pleasure: float, arousal: float, dominance: float) -> None:
        self.pleasure = pleasure
        self.arousal = arousal
        self.dominance = dominance

class DummyItem:
    """Dummy class to represent a context entry with content and optional properties."""
    def __init__(self, content: str, importance: float = 0.5, tags: list = None, emotional_state: Any = None) -> None:
        self.content = content
        self.importance = importance
        self.tags = tags or []
        self.emotional_state = emotional_state

def test_visualizer_initialization_and_formatting() -> None:
    """Tests that ContextEngineVisualizer formats the context correctly matching the OpenClaw schema."""
    visualizer = ContextEngineVisualizer(schema_version="openclaw_test")
    assert visualizer.schema_version == "openclaw_test"

    # Test formatting of empty/None context
    formatted_none = visualizer.format_context_state(user_id=42, new_context=None)
    assert formatted_none["schema_version"] == "openclaw_test"
    assert formatted_none["user_id"] == 42
    assert formatted_none["status"] == "updated"
    assert formatted_none["items"] == []
    assert formatted_none["item_count"] == 0

    # Test formatting of list with MemoryEntry / custom items
    emotion = DummyEmotionalState(0.1, 0.2, 0.3)
    item1 = DummyItem("First update", importance=0.8, tags=["tag1"], emotional_state=emotion)
    item2 = DummyItem("Second update", importance=0.4, tags=["tag2"])

    formatted_list = visualizer.format_context_state(user_id=42, new_context=[item1, item2], status="saved")
    assert formatted_list["status"] == "saved"
    assert formatted_list["item_count"] == 2
    assert len(formatted_list["items"]) == 2

    first_formatted = formatted_list["items"][0]
    assert first_formatted["content"] == "First update"
    assert first_formatted["importance"] == 0.8
    assert first_formatted["tags"] == ["tag1"]
    assert first_formatted["emotional_state"] == {"pleasure": 0.1, "arousal": 0.2, "dominance": 0.3}

    second_formatted = formatted_list["items"][1]
    assert second_formatted["content"] == "Second update"
    assert second_formatted["importance"] == 0.4
    assert second_formatted["tags"] == ["tag2"]
    assert "emotional_state" not in second_formatted

def test_subscriber_broadcasting() -> None:
    """Tests that ContextEngineVisualizer successfully triggers callbacks to subscribers."""
    visualizer = ContextEngineVisualizer()

    received_states = []
    def subscriber_callback(state: dict) -> None:
        received_states.append(state)

    visualizer.subscribe(subscriber_callback)

    # Trigger update
    visualizer.on_context_update("context update", user_id=123)

    assert len(received_states) == 1
    assert received_states[0]["user_id"] == 123
    assert received_states[0]["status"] == "updated"
    assert received_states[0]["items"][0]["content"] == "context update"

    # Trigger write
    visualizer.after_write("context save", user_id=123)
    assert len(received_states) == 2
    assert received_states[1]["status"] == "saved"
    assert received_states[1]["items"][0]["content"] == "context save"

    # Test unsubscribe
    visualizer.unsubscribe(subscriber_callback)
    visualizer.on_context_update("another update", user_id=123)
    assert len(received_states) == 2  # No new callback trigger

def test_json_export_and_retrieval() -> None:
    """Tests state retrieval and JSON string output matches OpenClaw schema constraints."""
    visualizer = ContextEngineVisualizer()
    user_id = 999

    # Initially empty state
    json_empty = visualizer.get_state_json(user_id)
    parsed_empty = json.loads(json_empty)
    assert parsed_empty["schema_version"] == "openclaw_v1"
    assert parsed_empty["user_id"] == user_id
    assert parsed_empty["status"] == "empty"
    assert parsed_empty["items"] == []
    assert parsed_empty["item_count"] == 0

    # Put state
    visualizer.on_context_update(["Entry A"], user_id=user_id)
    latest = visualizer.get_latest_state(user_id)
    assert latest is not None
    assert latest["item_count"] == 1

    json_populated = visualizer.get_state_json(user_id)
    parsed_populated = json.loads(json_populated)
    assert parsed_populated["user_id"] == user_id
    assert parsed_populated["status"] == "updated"
    assert parsed_populated["items"][0]["content"] == "Entry A"

@pytest.mark.asyncio
async def test_integration_with_context_engine() -> None:
    """Tests the visualizer integration with a standard ContextEngine as a plugin."""
    visualizer = ContextEngineVisualizer()
    engine = ContextEngine(plugins=[visualizer])

    received_states = []
    visualizer.subscribe(lambda state: received_states.append(state))

    # Trigger update on ContextEngine
    engine.update_context(["Engine context update"], user_id=777)
    assert len(received_states) == 1
    assert received_states[0]["user_id"] == 777
    assert received_states[0]["items"][0]["content"] == "Engine context update"

    # Trigger write on ContextEngine
    engine.write_context(["Engine context saved"], user_id=777)
    # write_context triggers both update_context (first update) and after_write (second update)
    assert len(received_states) == 3
    assert received_states[-1]["status"] == "saved"
    assert received_states[-1]["items"][0]["content"] == "Engine context saved"

@pytest.mark.asyncio
async def test_integration_with_context_engine_v5() -> None:
    """Tests the visualizer integration and dynamic unregistration with ContextEngineV5."""
    visualizer = ContextEngineVisualizer()
    engine = ContextEngineV5(plugins=[visualizer])

    received_states = []
    visualizer.subscribe(lambda state: received_states.append(state))

    # Trigger update
    engine.update_context(["V5 update"], user_id=888)
    assert len(received_states) == 1
    assert received_states[0]["items"][0]["content"] == "V5 update"

    # Unregister visualizer dynamically
    engine.unregister_plugin(visualizer)

    # Trigger update again
    engine.update_context(["Another V5 update"], user_id=888)
    assert len(received_states) == 1  # Should not have received the second update
