import pytest
from unittest.mock import Mock, MagicMock
from typing import Any
import json

from magda_agent.visualization.canvas import CanvasVisualizer
from magda_agent.consciousness.core import Consciousness

@pytest.fixture
def mock_consciousness() -> MagicMock:
    consciousness = MagicMock(spec=Consciousness)

    # Mock Emotions
    consciousness.emotions = MagicMock()
    consciousness.emotions.get_summary.return_value = "Feeling good"

    pad_state = MagicMock()
    pad_state.pleasure = 0.8
    pad_state.arousal = 0.5
    pad_state.dominance = 0.9
    consciousness.emotions.get_state_history.return_value = [pad_state]

    # Mock Mental States
    consciousness.mental_states = MagicMock()
    consciousness.mental_states.get_summary.return_value = "Focused"

    # Mock Memory
    consciousness.memory = MagicMock()
    consciousness.memory.get_summary.return_value = "Remembered 5 items"

    # Mock Skills
    consciousness.skills = MagicMock()
    consciousness.skills.skills = {"skill1": "desc", "skill2": "desc"}

    # Mock Planner
    consciousness.planner = MagicMock()
    consciousness.planner.get_state_summary.return_value = "Planning task X"

    return consciousness


def test_canvas_visualizer_formatting(mock_consciousness: MagicMock) -> None:
    """Test that CanvasVisualizer correctly formats the agent state."""
    visualizer = CanvasVisualizer(mock_consciousness)
    state = visualizer.get_formatted_state(user_id="user123")

    assert "emotions" in state
    assert state["emotions"]["summary"] == "Feeling good"
    assert state["emotions"]["pad"]["pleasure"] == 0.8

    assert "mental_states" in state
    assert state["mental_states"]["summary"] == "Focused"

    assert "memory" in state
    assert state["memory"]["summary"] == "Remembered 5 items"

    assert "skills" in state
    assert state["skills"] == ["skill1", "skill2"]

    assert "planner" in state
    assert state["planner"]["summary"] == "Planning task X"

    assert "error" not in state

def test_canvas_visualizer_json(mock_consciousness: MagicMock) -> None:
    """Test that CanvasVisualizer returns valid JSON."""
    visualizer = CanvasVisualizer(mock_consciousness)
    json_str = visualizer.get_state_json(user_id="user123")

    # Parse back to ensure valid JSON
    data = json.loads(json_str)
    assert data["emotions"]["summary"] == "Feeling good"

def test_canvas_visualizer_handles_errors(mock_consciousness: MagicMock) -> None:
    """Test that CanvasVisualizer handles exceptions gracefully."""
    # Force an error
    mock_consciousness.emotions.get_summary.side_effect = Exception("Test Error")

    visualizer = CanvasVisualizer(mock_consciousness)
    state = visualizer.get_formatted_state()

    assert "error" in state
    assert "Test Error" in state["error"]
