import pytest
from magda_agent.learning.dialogue_v3 import DialogueOnlineLearnerV3

def test_process_turn_preference() -> None:
    """Test extracting a preference from dialogue turn."""
    learner = DialogueOnlineLearnerV3()
    learner.process_turn("You must always reply in JSON format.")

    modifiers = learner.get_context_modifiers()
    assert "--- Immediate Context Modifications ---" in modifiers
    assert "The user strongly prefers" in modifiers
    assert "always reply in JSON format" in modifiers

def test_process_turn_constraint() -> None:
    """Test extracting a constraint from dialogue turn."""
    learner = DialogueOnlineLearnerV3()
    learner.process_turn("Please never use emojis.")

    modifiers = learner.get_context_modifiers()
    assert "--- Immediate Context Modifications ---" in modifiers
    assert "Constraint established" in modifiers
    assert "never use emojis" in modifiers

def test_process_turn_no_insight() -> None:
    """Test when no learning insight is extracted."""
    learner = DialogueOnlineLearnerV3()
    learner.process_turn("Hello, how are you?")

    modifiers = learner.get_context_modifiers()
    assert modifiers == ""
    assert len(learner.active_modifiers) == 0

def test_clear_session() -> None:
    """Test clearing the active modifiers session."""
    learner = DialogueOnlineLearnerV3()
    learner.process_turn("You must always reply in JSON format.")
    assert len(learner.active_modifiers) == 1

    learner.clear_session()
    assert len(learner.active_modifiers) == 0
    assert learner.get_context_modifiers() == ""
