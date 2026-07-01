"""
Tests for the OpenClawRLSignals module.
"""

import pytest
from magda_agent.learning.openclaw_rl_signals import OpenClawRLSignals

def test_process_signal() -> None:
    """Test that a signal is correctly processed and stored."""
    processor = OpenClawRLSignals()

    # Process a positive user reply
    signal1 = processor.process_signal(
        source="user_reply",
        content="That worked perfectly, thanks!",
        sentiment_score=0.8
    )

    assert signal1["source"] == "user_reply"
    assert signal1["reward"] == 0.8
    assert signal1["is_positive"] is True

    # Process a negative tool output
    signal2 = processor.process_signal(
        source="tool_output",
        content="Error: file not found.",
        sentiment_score=-0.9
    )

    assert signal2["source"] == "tool_output"
    assert signal2["reward"] == -0.9
    assert signal2["is_positive"] is False

def test_get_recent_signals() -> None:
    """Test retrieving recent signals respects the limit."""
    processor = OpenClawRLSignals()

    for i in range(15):
        processor.process_signal(
            source="user_reply",
            content=f"Message {i}",
            sentiment_score=0.1
        )

    recent = processor.get_recent_signals(limit=5)
    assert len(recent) == 5
    assert recent[-1]["content"] == "Message 14"
