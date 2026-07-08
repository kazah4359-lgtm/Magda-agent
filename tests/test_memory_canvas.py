import pytest
import json
from unittest.mock import MagicMock
from magda_agent.visualization.memory_canvas import MemoryCanvasVisualizer

def test_memory_canvas_visualizer_formatting():
    """Test that the MemoryCanvasVisualizer correctly formats the agent's memory state."""

    # Mock Memory System
    mock_memory_system = MagicMock()

    # Mock Working Memory
    mock_working = MagicMock()
    mock_working.get_entries.return_value = [1, 2, 3, 4]
    mock_working.get_summary.return_value = "Working memory summary content"
    mock_memory_system.working_memory = mock_working

    # Mock Episodic Memory
    mock_memory_system.episodic = MagicMock()

    # Mock Semantic Memory
    mock_memory_system.semantic = MagicMock()

    # Mock Procedural Memory
    mock_procedural = MagicMock()
    mock_procedural.skills = {"skill1": "impl1", "skill2": "impl2", "skill3": "impl3"}
    mock_memory_system.procedural = mock_procedural

    # Create visualizer and get state
    visualizer = MemoryCanvasVisualizer(mock_memory_system)
    state = visualizer.get_formatted_memory_state(user_id="123")

    # Assertions
    assert "working" in state
    assert state["working"]["count"] == 4
    assert state["working"]["summary"] == "Working memory summary content"

    assert "episodic" in state
    assert state["episodic"]["status"] == "active"

    assert "semantic" in state
    assert state["semantic"]["status"] == "active"

    assert "procedural" in state
    assert state["procedural"]["status"] == "active"
    assert state["procedural"]["skills_count"] == 3

    assert "error" not in state

    # Test JSON output
    json_str = visualizer.get_memory_state_json(user_id="123")
    json_data = json.loads(json_str)
    assert json_data == state

def test_memory_canvas_visualizer_empty_system():
    """Test handling of None memory system."""
    visualizer = MemoryCanvasVisualizer(None)
    state = visualizer.get_formatted_memory_state()

    assert state["working"] == {}
    assert state["episodic"] == {}
    assert state["semantic"] == {}
    assert state["procedural"] == {}
    assert "error" not in state

def test_memory_canvas_visualizer_missing_components():
    """Test handling when some memory components are missing."""
    mock_memory_system = MagicMock()
    mock_memory_system.working_memory = None
    mock_memory_system.episodic = None
    mock_memory_system.semantic = None
    mock_memory_system.procedural = None

    visualizer = MemoryCanvasVisualizer(mock_memory_system)
    state = visualizer.get_formatted_memory_state()

    assert state["working"] == {}
    assert state["episodic"] == {}
    assert state["semantic"] == {}
    assert state["procedural"] == {}
    assert "error" not in state
