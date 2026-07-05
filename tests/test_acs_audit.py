import pytest
from unittest.mock import MagicMock
from magda_agent.safety.acs_guard import ACSGuard, SecurityViolationError
from magda_agent.safety.audit_trail import AuditTrail
from magda_agent.safety.acs_audit import ACSAuditLogger

def test_acs_audit_logger_allowed() -> None:
    """Test that ACSAuditLogger logs an allowed action."""
    mock_acs_guard = MagicMock(spec=ACSGuard)
    mock_audit_trail = MagicMock(spec=AuditTrail)

    workflow_data = {"tool": "allowed_tool", "action": "read", "kwargs": {"id": 1}}

    # intercept_action should return the data when successful
    mock_acs_guard.intercept_action.return_value = workflow_data

    logger = ACSAuditLogger(acs_guard=mock_acs_guard, audit_trail=mock_audit_trail)
    result = logger.evaluate_and_intercept(workflow_data)

    assert result == workflow_data
    mock_acs_guard.intercept_action.assert_called_once_with(workflow_data)

    mock_audit_trail.log_call.assert_called_once()
    call_args = mock_audit_trail.log_call.call_args[1]
    assert call_args["tool_name"] == "allowed_tool"
    assert call_args["kwargs"] == {"id": 1}
    assert call_args["why"] == "ACSGuard evaluation passed"
    assert call_args["result"] == "allowed"
    assert "duration" in call_args

def test_acs_audit_logger_blocked() -> None:
    """Test that ACSAuditLogger logs a blocked action."""
    mock_acs_guard = MagicMock(spec=ACSGuard)
    mock_audit_trail = MagicMock(spec=AuditTrail)

    workflow_data = {"tool": "forbidden_tool", "action": "read", "kwargs": {"id": 2}}

    # intercept_action should raise SecurityViolationError when blocked
    error_msg = "Action blocked by ACS checkpoint 3: Tool policy failed"
    mock_acs_guard.intercept_action.side_effect = SecurityViolationError(error_msg)

    logger = ACSAuditLogger(acs_guard=mock_acs_guard, audit_trail=mock_audit_trail)

    with pytest.raises(SecurityViolationError, match="Action blocked by ACS checkpoint 3"):
        logger.evaluate_and_intercept(workflow_data)

    mock_acs_guard.intercept_action.assert_called_once_with(workflow_data)

    mock_audit_trail.log_call.assert_called_once()
    call_args = mock_audit_trail.log_call.call_args[1]
    assert call_args["tool_name"] == "forbidden_tool"
    assert call_args["kwargs"] == {"id": 2}
    assert call_args["why"] == f"ACSGuard evaluation failed: {error_msg}"
    assert call_args["result"] == "blocked"
    assert "duration" in call_args
