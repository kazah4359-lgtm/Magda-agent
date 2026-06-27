import pytest
from magda_agent.learning.interactive_rl import InteractiveLearner


@pytest.fixture
def learner():
    return InteractiveLearner()


def test_analyze_signal_positive(learner):
    """Test positive reward extraction."""
    assert learner.analyze_signal("This is great!") == 1.0
    assert learner.analyze_signal("Yes, thanks for that.") == 1.0
    assert learner.analyze_signal("good job.") == 1.0


def test_analyze_signal_negative(learner):
    """Test negative reward extraction."""
    assert learner.analyze_signal("This is wrong.") == -1.0
    assert learner.analyze_signal("No, terrible idea.") == -1.0
    assert learner.analyze_signal("bad result.") == -1.0


def test_analyze_signal_neutral(learner):
    """Test neutral reward extraction."""
    assert learner.analyze_signal("I am here.") == 0.0
    assert learner.analyze_signal("What time is it?") == 0.0


def test_analyze_signal_empty(learner):
    """Test empty string reward extraction."""
    assert learner.analyze_signal("") == 0.0


@pytest.mark.asyncio
async def test_process_interaction_positive(learner):
    """Test state update with positive reply."""
    await learner.process_interaction("This is great!", "coding_skill")
    state = learner.get_state()
    assert "coding_skill" in state
    assert state["coding_skill"] == 1.2  # 1.0 (default) + 0.2 (1.0 * 0.2)


@pytest.mark.asyncio
async def test_process_interaction_negative(learner):
    """Test state update with negative reply."""
    await learner.process_interaction("This is terrible.", "math_skill")
    state = learner.get_state()
    assert "math_skill" in state
    assert state["math_skill"] == 0.8  # 1.0 (default) - 0.2 (1.0 * 0.2)


@pytest.mark.asyncio
async def test_process_interaction_neutral(learner):
    """Test state update with neutral reply."""
    await learner.process_interaction("Okay.", "search_skill")
    state = learner.get_state()
    assert "search_skill" in state
    assert state["search_skill"] == 1.0  # 1.0 (default) + 0.0 (0.0 * 0.2)


@pytest.mark.asyncio
async def test_process_interaction_clamp(learner):
    """Test state clamping to minimum value."""
    # Force state to go below minimum
    learner.learning_state["bad_skill"] = 0.2
    await learner.process_interaction("This is bad.", "bad_skill")
    state = learner.get_state()
    assert state["bad_skill"] == 0.1  # Clamped to 0.1


@pytest.mark.asyncio
async def test_process_interaction_empty_reply(learner):
    """Test state update with empty reply."""
    await learner.process_interaction("", "empty_skill")
    state = learner.get_state()
    assert "empty_skill" not in state

def test_analyze_signal_positive_new_keywords(learner):
    """Test positive reward extraction with new keywords."""
    assert learner.analyze_signal("This is awesome!") == 1.0
    assert learner.analyze_signal("excellent job.") == 1.0


def test_analyze_signal_negative_new_keywords(learner):
    """Test negative reward extraction with new keywords."""
    assert learner.analyze_signal("This is awful.") == -1.0
    assert learner.analyze_signal("horrible idea.") == -1.0


@pytest.mark.asyncio
async def test_process_batch_interactions(learner):
    """Test process_batch_interactions method."""
    interactions = [
        ("This is great!", "skill_a"),
        ("This is terrible.", "skill_b"),
        ("Okay.", "skill_c"),
    ]
    await learner.process_batch_interactions(interactions)
    state = learner.get_state()

    assert "skill_a" in state
    assert state["skill_a"] == 1.2

    assert "skill_b" in state
    assert state["skill_b"] == 0.8

    assert "skill_c" in state
    assert state["skill_c"] == 1.0
