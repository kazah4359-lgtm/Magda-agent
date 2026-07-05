import pytest
from magda_agent.learning.openclaw_rl_v3 import OnlineRLFeedbackLoop

def test_map_reply_to_reward_positive() -> None:
    """Tests parsing positive user replies to reward scores."""
    loop = OnlineRLFeedbackLoop()
    assert loop._map_reply_to_reward("That's great") == 1.0
    assert loop._map_reply_to_reward("good job") == 1.0
    assert loop._map_reply_to_reward("awesome!") == 1.0
    assert loop._map_reply_to_reward("Yes please") == 1.0
    assert loop._map_reply_to_reward("thanks") == 1.0

def test_map_reply_to_reward_negative() -> None:
    """Tests parsing negative user replies to reward scores."""
    loop = OnlineRLFeedbackLoop()
    assert loop._map_reply_to_reward("That's bad") == -1.0
    assert loop._map_reply_to_reward("terrible") == -1.0
    assert loop._map_reply_to_reward("no") == -1.0
    assert loop._map_reply_to_reward("wrong") == -1.0

def test_map_reply_to_reward_neutral() -> None:
    """Tests parsing neutral user replies to reward scores."""
    loop = OnlineRLFeedbackLoop()
    assert loop._map_reply_to_reward("Okay") == 0.0
    assert loop._map_reply_to_reward("Maybe") == 0.0
    assert loop._map_reply_to_reward("I don't know") == 0.0

def test_process_feedback() -> None:
    """Tests processing feedback and verifying Q-value updates."""
    loop = OnlineRLFeedbackLoop()

    # Test positive feedback
    loop.process_feedback("skill_1", "Great")
    assert loop.get_q_value("skill_1") == 0.1  # 0 + 0.1 * (1 - 0)

    # Test multiple positive feedback
    loop.process_feedback("skill_1", "Good")
    assert loop.get_q_value("skill_1") == 0.19 # 0.1 + 0.1 * (1 - 0.1)

    # Test negative feedback
    loop.process_feedback("skill_2", "Bad")
    assert loop.get_q_value("skill_2") == -0.1 # 0 + 0.1 * (-1 - 0)
