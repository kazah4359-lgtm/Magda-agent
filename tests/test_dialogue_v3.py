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

def test_process_turn_positive_reinforcement() -> None:
    """Test positive reinforcement from user message."""
    learner = DialogueOnlineLearnerV3()
    # Trigger positive sentiment
    for _ in range(5):
        learner.process_turn("This is great! I am so happy with your work!")

    assert learner.weights["verbosity"] > 1.0
    assert learner.weights["empathy"] > 1.0

    modifiers = learner.get_context_modifiers()
    assert "--- Dynamic Behavior Adjustments ---" in modifiers
    assert "Increase verbosity" in modifiers
    assert "Increase empathy" in modifiers

def test_process_turn_negative_reinforcement() -> None:
    """Test negative reinforcement from user message."""
    learner = DialogueOnlineLearnerV3()
    # Trigger negative sentiment
    for _ in range(5):
        learner.process_turn("This is terrible and bad. I am very angry.")

    assert learner.weights["verbosity"] < 1.0
    assert learner.weights["directness"] > 1.0

    modifiers = learner.get_context_modifiers()
    assert "--- Dynamic Behavior Adjustments ---" in modifiers
    assert "Decrease verbosity" in modifiers
    assert "Increase directness" in modifiers

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
    learner.process_turn("Great job!")
    assert len(learner.active_modifiers) == 1
    assert learner.weights["verbosity"] > 1.0

    learner.clear_session()
    assert len(learner.active_modifiers) == 0
    assert learner.weights["verbosity"] == 1.0
    assert learner.get_context_modifiers() == ""
