import pytest
import json
from unittest.mock import MagicMock
from magda_agent.visualization.canvas import CanvasVisualizer

class DummyPADState:
    def __init__(self, p, a, d):
        self.pleasure = p
        self.arousal = a
        self.dominance = d

def test_canvas_visualizer_formatting():
    """Test that the CanvasVisualizer correctly formats the agent state."""

    # Mock the consciousness and its components
    mock_consciousness = MagicMock()

    # Mock Emotions
    mock_emotions = MagicMock()
    mock_emotions.get_summary.return_value = "Feeling Neutral"
    mock_emotions.get_state_history.return_value = [DummyPADState(0.1, 0.2, 0.3)]
    mock_consciousness.emotions = mock_emotions

    # Mock Mental States
    mock_mental_states = MagicMock()
    mock_mental_states.get_summary.return_value = "Focused"
    mock_consciousness.mental_states = mock_mental_states

    # Mock Memory
    mock_memory = MagicMock()
    mock_memory.get_summary.return_value = "Working memory: 5 items"
    mock_consciousness.memory = mock_memory

    # Mock Skills
    mock_skills = MagicMock()
    mock_skills.skills = {"skill_a": None, "skill_b": None}
    mock_consciousness.skills = mock_skills

    # Mock Planner
    mock_planner = MagicMock()
    mock_planner.get_state_summary.return_value = "Current plan: None"
    mock_consciousness.planner = mock_planner

    # Create visualizer and get state
    visualizer = CanvasVisualizer(mock_consciousness)
    state = visualizer.get_formatted_state(user_id="test_user")

    # Assertions
    assert "emotions" in state
    assert state["emotions"]["summary"] == "Feeling Neutral"
    assert state["emotions"]["pad"]["pleasure"] == 0.1
    assert state["emotions"]["pad"]["arousal"] == 0.2
    assert state["emotions"]["pad"]["dominance"] == 0.3

    assert "mental_states" in state
    assert state["mental_states"]["summary"] == "Focused"

    assert "memory" in state
    assert state["memory"]["summary"] == "Working memory: 5 items"

    assert "skills" in state
    assert set(state["skills"]) == {"skill_a", "skill_b"}

    assert "planner" in state
    assert state["planner"]["summary"] == "Current plan: None"

    # Test JSON output
    json_str = visualizer.get_state_json(user_id="test_user")
    json_data = json.loads(json_str)
    assert json_data == state

def test_canvas_visualizer_error_handling():
    """Test that CanvasVisualizer handles missing components gracefully."""
    mock_consciousness = MagicMock()

    # Set components to None
    mock_consciousness.emotions = None
    mock_consciousness.mental_states = None
    mock_consciousness.memory = None
    mock_consciousness.skills = None
    mock_consciousness.planner = None

    visualizer = CanvasVisualizer(mock_consciousness)
    state = visualizer.get_formatted_state()

    assert state["emotions"] == {}
    assert state["mental_states"] == {}
    assert state["memory"] == {}
    assert state["skills"] == []
    assert state["planner"] == {}
    assert "error" not in state
