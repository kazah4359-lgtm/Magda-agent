import pytest
from typing import Any, Dict
from magda_agent.learning.interactive_feedback import InteractiveFeedbackFormatter
from magda_agent.learning.interactive_feedback import InteractiveFeedbackFormatterV2

def test_parse_corrections_explicit_score() -> None:
    """Tests if explicit score is correctly extracted."""
    formatter = InteractiveFeedbackFormatter()

    result = formatter.parse_corrections("This is okay, maybe 7.5/10.")
    assert result["explicit_score"] == 7.5

    result2 = formatter.parse_corrections("I give it 10 / 10!")
    assert result2["explicit_score"] == 10.0

    result3 = formatter.parse_corrections("Score is 11/10")
    # Should not capture out of bounds, though regex captures it, our logic discards if > 10.
    assert result3["explicit_score"] is None

def test_parse_corrections_implicit_sentiment() -> None:
    """Tests if implicit sentiment is correctly calculated."""
    formatter = InteractiveFeedbackFormatter()

    result = formatter.parse_corrections("Good job, thanks!")
    # 'good', 'thanks' -> 2 positive
    assert result["implicit_sentiment"] == 1.0

    result2 = formatter.parse_corrections("This is terrible and bad.")
    # 'terrible', 'bad' -> 2 negative
    assert result2["implicit_sentiment"] == -1.0

    result3 = formatter.parse_corrections("It is good but bad")
    # 1 positive, 1 negative -> 0.0
    assert result3["implicit_sentiment"] == 0.0

def test_parse_corrections_is_correction() -> None:
    """Tests if corrections are correctly detected."""
    formatter = InteractiveFeedbackFormatter()

    result = formatter.parse_corrections("No, actually it's this.")
    assert result["is_correction"] is True

    result2 = formatter.parse_corrections("Please fix the typo.")
    assert result2["is_correction"] is True

    result3 = formatter.parse_corrections("Great, thanks!")
    assert result3["is_correction"] is False

def test_parse_corrections_empty() -> None:
    """Tests behavior with empty string."""
    formatter = InteractiveFeedbackFormatter()
    result = formatter.parse_corrections("")

    assert result["is_correction"] is False
    assert result["explicit_score"] is None
    assert result["implicit_sentiment"] == 0.0


def test_parse_detailed_feedback_praise() -> None:
    """Tests if detailed feedback correctly identifies praise intent."""
    formatter = InteractiveFeedbackFormatterV2()

    result = formatter.parse_detailed_feedback("This is great! 9/10.")
    assert result["intent"] == "praise"
    assert result["explicit_score"] == 9.0
    assert result["implicit_sentiment"] > 0

def test_parse_detailed_feedback_criticism() -> None:
    """Tests if detailed feedback correctly identifies criticism intent."""
    formatter = InteractiveFeedbackFormatterV2()

    result = formatter.parse_detailed_feedback("No, that is completely wrong.")
    assert result["intent"] == "criticism"
    assert result["is_correction"] is True
    assert result["implicit_sentiment"] < 0

def test_parse_detailed_feedback_neutral() -> None:
    """Tests if detailed feedback correctly identifies neutral intent."""
    formatter = InteractiveFeedbackFormatterV2()

    result = formatter.parse_detailed_feedback("Here is some random text.")
    assert result["intent"] == "neutral"
    assert result["is_correction"] is False
    assert result["implicit_sentiment"] == 0.0
