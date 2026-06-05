import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.metacognition.failure_patterns import FailurePatternTracker

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="Always double check variables before referencing them.")
    return llm

@pytest.fixture
def tracker(mock_llm):
    # Use memory database
    return FailurePatternTracker(llm=mock_llm, db_path=":memory:")

def test_log_failure_and_get_by_category(tracker):
    tracker.log_failure("pytest", "AssertionError in test_foo")
    tracker.log_failure("pytest", "ModuleNotFoundError in test_bar")
    tracker.log_failure("lint", "E501 line too long")

    failures = tracker._get_failures_by_category("pytest")
    assert len(failures) == 2
    assert "AssertionError in test_foo" in failures
    assert "ModuleNotFoundError in test_bar" in failures

    lint_failures = tracker._get_failures_by_category("lint")
    assert len(lint_failures) == 1
    assert lint_failures[0] == "E501 line too long"

@pytest.mark.asyncio
async def test_detect_recurring_patterns(tracker, mock_llm):
    # Log 3 failures in 'pytest' (threshold is 2)
    tracker.log_failure("pytest", "Error 1")
    tracker.log_failure("pytest", "Error 2")
    tracker.log_failure("pytest", "Error 3")

    # Log 2 failures in 'lint' (won't trigger pattern detection)
    tracker.log_failure("lint", "Error A")
    tracker.log_failure("lint", "Error B")

    results = await tracker.detect_recurring_patterns(threshold=2)

    assert len(results) == 1
    assert results[0]["category"] == "pytest"
    assert results[0]["rule"] == "Always double check variables before referencing them."

    mock_llm.chat_completion.assert_called_once()

    rules = tracker.get_preventive_rules()
    assert len(rules) == 1
    assert rules[0] == "Always double check variables before referencing them."

@pytest.mark.asyncio
async def test_detect_recurring_patterns_no_duplicate_rules(tracker, mock_llm):
    tracker.log_failure("pytest", "Error 1")
    tracker.log_failure("pytest", "Error 2")
    tracker.log_failure("pytest", "Error 3")

    results1 = await tracker.detect_recurring_patterns(threshold=2)
    assert len(results1) == 1

    # Call again immediately, should not generate another rule for the same category
    results2 = await tracker.detect_recurring_patterns(threshold=2)
    assert len(results2) == 0

    # LLM should have only been called once
    mock_llm.chat_completion.assert_called_once()
