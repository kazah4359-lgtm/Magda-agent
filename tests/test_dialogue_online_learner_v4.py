import pytest
from magda_agent.learning.dialogue_online_learner_v4 import DialogueOnlineLearnerV4

def test_process_turn_preference() -> None:
    """Test extracting a preference from dialogue turn."""
    learner = DialogueOnlineLearnerV4()
    learner.process_turn("You must always reply in JSON format.")

    modifiers = learner.get_context_modifiers()
    assert "--- Immediate Context Modifications ---" in modifiers
    assert "The user strongly prefers" in modifiers
    assert "always reply in JSON format" in modifiers

def test_process_turn_constraint() -> None:
    """Test extracting a constraint from dialogue turn."""
    learner = DialogueOnlineLearnerV4()
    learner.process_turn("Please never use emojis.")

    modifiers = learner.get_context_modifiers()
    assert "--- Immediate Context Modifications ---" in modifiers
    assert "Constraint established" in modifiers
    assert "never use emojis" in modifiers

def test_process_turn_positive_reinforcement() -> None:
    """Test positive reinforcement from user message."""
    learner = DialogueOnlineLearnerV4()
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
    learner = DialogueOnlineLearnerV4()
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
    learner = DialogueOnlineLearnerV4()
    learner.process_turn("Hello, how are you?")

    modifiers = learner.get_context_modifiers()
    assert modifiers == ""
    assert len(learner.active_modifiers) == 0

def test_capture_and_process_state_action_reward() -> None:
    """Test state-action capturing and reward logging."""
    learner = DialogueOnlineLearnerV4()

    # 1. Capture state and action
    learner.capture_state_action("User asked for a joke", "Here is a joke: Why did the chicken cross the road?")
    assert learner.last_state_action is not None
    assert learner.last_state_action["state"] == "User asked for a joke"
    assert learner.last_state_action["action"] == "Here is a joke: Why did the chicken cross the road?"
    assert len(learner.trajectory_log) == 0

    # 2. Process next-state signal with positive sentiment
    learner.process_turn("Haha, that is a great joke!")

    # The last state action should be consumed
    assert learner.last_state_action is None
    # We should have a trajectory logged with positive reward
    assert len(learner.trajectory_log) == 1
    trajectory = learner.trajectory_log[0]
    assert trajectory["state"] == "User asked for a joke"
    assert trajectory["action"] == "Here is a joke: Why did the chicken cross the road?"
    assert trajectory["next_state"] == "Haha, that is a great joke!"
    assert trajectory["reward"] > 0

def test_clear_session() -> None:
    """Test clearing the active modifiers session."""
    learner = DialogueOnlineLearnerV4()
    learner.process_turn("You must always reply in JSON format.")
    learner.process_turn("Great job!")
    learner.capture_state_action("state", "action")

    assert len(learner.active_modifiers) == 1
    assert learner.weights["verbosity"] > 1.0
    assert learner.last_state_action is not None

    learner.clear_session()
    assert len(learner.active_modifiers) == 0
    assert learner.weights["verbosity"] == 1.0
    assert learner.get_context_modifiers() == ""
    assert learner.last_state_action is None
    assert len(learner.trajectory_log) == 0
