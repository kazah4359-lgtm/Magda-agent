import pytest
import json
from unittest.mock import MagicMock
from magda_agent.visualization.canvas_v3 import CanvasVisualizerV3

class DummyPADState:
    def __init__(self, p, a, d):
        self.pleasure = p
        self.arousal = a
        self.dominance = d

def test_canvas_visualizer_v3_formatting():
    """Test that the CanvasVisualizerV3 correctly formats the complex agent state."""

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
    mock_working = MagicMock()
    mock_working.items = [1, 2, 3]
    mock_working.get_summary.return_value = "Working memory summary"
    mock_memory.working_memory = mock_working
    mock_memory.episodic = MagicMock()
    mock_memory.semantic = MagicMock()
    mock_procedural = MagicMock()
    mock_procedural.skills = {"skill1": None, "skill2": None}
    mock_memory.procedural = mock_procedural
    mock_consciousness.memory = mock_memory

    # Mock Skills
    mock_skills = MagicMock()
    mock_skills.skills = {"skill_a": None, "skill_b": None}
    mock_consciousness.skills = mock_skills

    # Mock Planner
    mock_planner = MagicMock()
    mock_planner.get_state_summary.return_value = "Current plan: Execute tasks"
    mock_plan = MagicMock()
    mock_plan.goal = "Test Goal"
    mock_plan.steps = [1, 2]
    mock_plan.dependencies = ["dep1", "dep2"]
    mock_planner.current_plan = mock_plan
    mock_consciousness.planner = mock_planner

    # Mock Drives
    mock_hypothalamus = MagicMock()
    mock_hypothalamus.energy = 0.8
    mock_hypothalamus.boredom = 0.2
    mock_hypothalamus.get_summary.return_value = "High energy, low boredom"
    mock_consciousness.hypothalamus = mock_hypothalamus

    # Mock Global Workspace
    mock_workspace = MagicMock()
    mock_workspace.focused_event = "User Input"
    mock_consciousness.global_workspace = mock_workspace

    # Mock RL Metrics
    mock_rl_metrics = MagicMock()
    mock_rl_metrics.get_visualization_data.return_value = {"status": "active", "q_values": {}}
    mock_consciousness.openclaw_rl_metrics = mock_rl_metrics

    visualizer = CanvasVisualizerV3(mock_consciousness)
    state = visualizer.get_formatted_state(user_id="test_user")

    assert state["emotions"]["summary"] == "Feeling Neutral"
    assert state["emotions"]["pad"]["pleasure"] == 0.1

    assert state["mental_states"]["summary"] == "Focused"

    assert state["memory"]["working"]["count"] == 3
    assert state["memory"]["working"]["summary"] == "Working memory summary"
    assert state["memory"]["episodic"]["status"] == "active"
    assert state["memory"]["semantic"]["status"] == "active"
    assert state["memory"]["procedural"]["status"] == "active"
    assert state["memory"]["procedural"]["skills_count"] == 2

    assert set(state["skills"]) == {"skill_a", "skill_b"}

    assert state["planner"]["summary"] == "Current plan: Execute tasks"
    assert state["planner"]["current_plan"]["goal"] == "Test Goal"
    assert state["planner"]["current_plan"]["step_count"] == 2
    assert state["planner"]["current_plan"]["dependencies"] == ["dep1", "dep2"]

    assert state["drives"]["energy"] == 0.8
    assert state["drives"]["boredom"] == 0.2
    assert state["drives"]["summary"] == "High energy, low boredom"

    assert state["global_workspace"]["focused_event"] == "User Input"
    assert state["global_workspace"]["active"] is True

    assert state["rl_metrics"]["status"] == "active"

    assert "error" not in state

    json_str = visualizer.get_state_json(user_id="test_user")
    json_data = json.loads(json_str)
    assert json_data == state


def test_canvas_visualizer_v3_error_handling():
    """Test that CanvasVisualizerV3 handles missing components gracefully."""
    mock_consciousness = MagicMock()
    mock_consciousness.emotions = None
    mock_consciousness.mental_states = None
    mock_consciousness.memory = None
    mock_consciousness.skills = None
    mock_consciousness.planner = None
    mock_consciousness.hypothalamus = None
    mock_consciousness.global_workspace = None
    mock_consciousness.openclaw_rl_metrics = None

    visualizer = CanvasVisualizerV3(mock_consciousness)
    state = visualizer.get_formatted_state()

    assert state["emotions"] == {}
    assert state["mental_states"] == {}
    assert state["memory"] == {"working": {}, "episodic": {}, "semantic": {}, "procedural": {}}
    assert state["skills"] == []
    assert state["planner"] == {}
    assert state["drives"] == {}
    assert state["global_workspace"] == {}
    assert "error" not in state
