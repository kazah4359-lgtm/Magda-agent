"""Tests for the ACS Taint Tracking Checkpoint v2 module."""

import pytest
from magda_agent.safety.acs_taint_v2 import OutputTaintTrackingCheckpointV2
from magda_agent.safety.taint import mark_tainted


def test_validate_clean_output():
    """Test validation with clean output."""
    checkpoint = OutputTaintTrackingCheckpointV2()
    workflow_data = {"output": "This is a clean output string."}

    passed, reason = checkpoint.validate(workflow_data)

    assert passed is True
    assert reason == "Output taint tracking passed."


def test_validate_tainted_output():
    """Test validation with tainted output."""
    checkpoint = OutputTaintTrackingCheckpointV2()
    workflow_data = {"output": mark_tainted("This is a tainted output string.")}

    passed, reason = checkpoint.validate(workflow_data)

    assert passed is False
    assert reason == "Output taint tracking failed: tainted data detected in output."


def test_validate_tainted_output_in_dict():
    """Test validation with tainted output inside a dictionary."""
    checkpoint = OutputTaintTrackingCheckpointV2()
    workflow_data = {"output": {"result": mark_tainted("Tainted result"), "status": "ok"}}

    passed, reason = checkpoint.validate(workflow_data)

    assert passed is False
    assert reason == "Output taint tracking failed: tainted data detected in output."


def test_validate_no_output():
    """Test validation when no output is present in workflow data."""
    checkpoint = OutputTaintTrackingCheckpointV2()
    workflow_data = {"action": "some_action"}

    passed, reason = checkpoint.validate(workflow_data)

    assert passed is True
    assert reason == "Output taint tracking passed: no output to validate."
