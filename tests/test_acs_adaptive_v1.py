import pytest
from unittest.mock import MagicMock
from magda_agent.safety.acs_adaptive_v1 import AdaptiveGuardrail
from magda_agent.safety.acs_checkpoints import ACSCheckpoints, CheckpointStage

@pytest.fixture
def mock_acs_checkpoints() -> MagicMock:
    """Fixture that provides a mock ACSCheckpoints instance."""
    mock = MagicMock(spec=ACSCheckpoints)
    # Default mock behavior: all stages pass
    mock._run_stage.return_value = (True, "Stage passed.")
    return mock

@pytest.fixture
def adaptive_guardrail(mock_acs_checkpoints: MagicMock) -> AdaptiveGuardrail:
    """Fixture that provides an AdaptiveGuardrail instance configured with the mock."""
    return AdaptiveGuardrail(acs_checkpoints=mock_acs_checkpoints)

def test_low_risk_bypasses_checkpoints(adaptive_guardrail: AdaptiveGuardrail, mock_acs_checkpoints: MagicMock) -> None:
    """Tests that low risk workflows only execute the INPUT stage."""
    workflow_data = {"action_name": "chat", "tool_name": "test_tool"}

    passed, msg = adaptive_guardrail.evaluate(workflow_data, risk_score="low")

    assert passed is True
    assert "Adaptive guardrails passed" in msg
    mock_acs_checkpoints._run_stage.assert_called_once_with(CheckpointStage.INPUT, workflow_data)

def test_medium_risk_runs_execution(adaptive_guardrail: AdaptiveGuardrail, mock_acs_checkpoints: MagicMock) -> None:
    """Tests that medium risk workflows execute both INPUT and EXECUTION stages, handling failures."""
    workflow_data = {"action_name": "chat", "tool_name": "forbidden_tool"}

    # Simulate EXECUTION stage failing
    def mock_run_stage(stage, data):
        if stage == CheckpointStage.EXECUTION:
            return False, "Checkpoint 3 Failed: Tool forbidden"
        return True, "Passed"

    mock_acs_checkpoints._run_stage.side_effect = mock_run_stage

    passed, msg = adaptive_guardrail.evaluate(workflow_data, risk_score="medium")

    assert passed is False
    assert "Checkpoint 3 Failed: Tool forbidden" in msg
    assert mock_acs_checkpoints._run_stage.call_count == 2

def test_high_risk_runs_all(adaptive_guardrail: AdaptiveGuardrail, mock_acs_checkpoints: MagicMock) -> None:
    """Tests that high risk workflows execute all stages, and properly catches output sanitization failures."""
    workflow_data = {"action_name": "chat", "tool_name": "valid_tool", "output": "secret"}

    # Simulate OUTPUT stage failing
    def mock_run_stage(stage, data):
        if stage == CheckpointStage.OUTPUT:
            return False, "Checkpoint 5 Failed: Sensitive data"
        return True, "Passed"

    mock_acs_checkpoints._run_stage.side_effect = mock_run_stage

    passed, msg = adaptive_guardrail.evaluate(workflow_data, risk_score="high")

    assert passed is False
    assert "Checkpoint 5 Failed: Sensitive data" in msg
    assert mock_acs_checkpoints._run_stage.call_count == 3
