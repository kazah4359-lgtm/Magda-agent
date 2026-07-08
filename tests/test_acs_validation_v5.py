import pytest
from magda_agent.safety.acs_validation_v5 import ACSValidationV5


@pytest.fixture
def acs_validator() -> ACSValidationV5:
    """Fixture to provide an instance of ACSValidationV5."""
    return ACSValidationV5()


def test_acs_validation_v5_safe_tool(acs_validator: ACSValidationV5) -> None:
    """Tests that a safe tool call passes validation."""
    workflow_data = {
        "tool": "read_file",
        "kwargs": {"filepath": "config.yaml"}
    }
    passed, reason = acs_validator.validate_tool_call(workflow_data)
    assert passed is True
    assert "Validation passed" in reason

    workflow_data_safe_execute = {
        "tool": "system_execute_code",
        "kwargs": {"command": "ls -l"}
    }
    passed, reason = acs_validator.validate_tool_call(workflow_data_safe_execute)
    assert passed is True
    assert "Validation passed" in reason


def test_acs_validation_v5_destructive_tool(acs_validator: ACSValidationV5) -> None:
    """Tests that potentially destructive tool calls are blocked."""
    # Test blocked tool
    workflow_data_blocked_tool = {
        "tool": "rm",
        "kwargs": {"target": "important_file.txt"}
    }
    passed, reason = acs_validator.validate_tool_call(workflow_data_blocked_tool)
    assert passed is False
    assert "is considered destructive and blocked" in reason

    # Test dangerous pattern in execute tool
    workflow_data_dangerous_pattern = {
        "tool": "run_in_bash_session",
        "kwargs": {"command": "rm -rf /"}
    }
    passed, reason = acs_validator.validate_tool_call(workflow_data_dangerous_pattern)
    assert passed is False
    assert "contains dangerous pattern 'rm -rf'" in reason

    # Test dangerous pattern in code payload
    workflow_data_dangerous_code = {
        "tool": "system_execute_code",
        "kwargs": {"code": "DROP TABLE users;"}
    }
    passed, reason = acs_validator.validate_tool_call(workflow_data_dangerous_code)
    assert passed is False
    assert "contains dangerous pattern 'drop table'" in reason


def test_acs_validation_v5_invalid_input(acs_validator: ACSValidationV5) -> None:
    """Tests validation with missing or invalid input."""
    # Missing tool
    workflow_data_missing_tool = {
        "kwargs": {"path": "/"}
    }
    passed, reason = acs_validator.validate_tool_call(workflow_data_missing_tool)
    assert passed is False
    assert "'tool' field is missing or empty" in reason

    # Invalid kwargs
    workflow_data_invalid_kwargs = {
        "tool": "ls",
        "kwargs": "not a dict"
    }
    passed, reason = acs_validator.validate_tool_call(workflow_data_invalid_kwargs)
    assert passed is False
    assert "'kwargs' must be a dictionary" in reason
