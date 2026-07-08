"""Tests for the MCP Kernel Taint Tracking Policy Engine."""

import pytest

from magda_agent.safety.mcp_taint_policy import MCPTaintPolicyEngine
from magda_agent.safety.taint_tracking_v2 import PolicyViolationError, TaintTrackerV2


def test_evaluate_stream_blocks_tainted_input_when_sensitive() -> None:
    """Test that evaluate_stream raises PolicyViolationError when sensitive and inputs are tainted."""
    tracker = TaintTrackerV2()
    engine = MCPTaintPolicyEngine(tracker=tracker)

    tainted_input = tracker.taint({"data": "untrusted"}, "malicious_source")

    with pytest.raises(PolicyViolationError, match="Tainted input to sensitive tool call"):
        engine.evaluate_stream(inputs=tainted_input, is_sensitive=True)


def test_evaluate_stream_allows_tainted_input_when_not_sensitive() -> None:
    """Test that evaluate_stream allows execution when inputs are tainted but tool is not sensitive."""
    tracker = TaintTrackerV2()
    engine = MCPTaintPolicyEngine(tracker=tracker)

    tainted_input = tracker.taint({"data": "untrusted"}, "malicious_source")

    # Should not raise an exception
    engine.evaluate_stream(inputs=tainted_input, is_sensitive=False)


def test_evaluate_stream_allows_clean_input_when_sensitive() -> None:
    """Test that evaluate_stream allows execution for a sensitive tool when inputs are not tainted."""
    tracker = TaintTrackerV2()
    engine = MCPTaintPolicyEngine(tracker=tracker)

    clean_input = {"data": "trusted"}

    # Should not raise an exception
    engine.evaluate_stream(inputs=clean_input, is_sensitive=True)


def test_evaluate_stream_allows_clean_input_when_not_sensitive() -> None:
    """Test that evaluate_stream allows execution for a non-sensitive tool when inputs are clean."""
    engine = MCPTaintPolicyEngine()

    clean_input = {"data": "trusted"}

    # Should not raise an exception
    engine.evaluate_stream(inputs=clean_input, is_sensitive=False)
