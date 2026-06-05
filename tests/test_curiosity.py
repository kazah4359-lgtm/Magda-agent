import pytest
from magda_agent.exploration.curiosity import CuriosityExplorer

def test_curiosity_initialization():
    """Test that CuriosityExplorer initializes with default and custom thresholds."""
    explorer_default = CuriosityExplorer()
    assert explorer_default.boredom_threshold == 0.8

    explorer_custom = CuriosityExplorer(boredom_threshold=0.5)
    assert explorer_custom.boredom_threshold == 0.5

def test_should_explore():
    """Test the should_explore boundary conditions."""
    explorer = CuriosityExplorer(boredom_threshold=0.8)

    # Below threshold
    assert not explorer.should_explore(0.79)
    assert not explorer.should_explore(0.0)

    # At threshold
    assert explorer.should_explore(0.8)

    # Above threshold
    assert explorer.should_explore(0.9)
    assert explorer.should_explore(1.0)

def test_explore_returns_safe_actions():
    """Test that explore returns a list of read-only string tasks."""
    explorer = CuriosityExplorer()
    actions = explorer.explore()

    assert isinstance(actions, list)
    assert len(actions) > 0
    assert all(isinstance(action, str) for action in actions)

    # Ensure some known safe keywords are present
    combined_actions = " ".join(actions).lower()
    assert "read" in combined_actions or "analyze" in combined_actions or "check" in combined_actions

def test_explore_with_mock_context():
    """Test explore with a mock context object."""
    explorer = CuriosityExplorer()
    mock_context = {"active_files": ["main.py"]}

    actions = explorer.explore(workspace_context=mock_context)
    assert len(actions) > 0
