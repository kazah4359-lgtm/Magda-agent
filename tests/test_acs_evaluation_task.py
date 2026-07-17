import pytest
from unittest.mock import MagicMock, patch

from magda_agent.scheduler.acs_evaluation_task import ACSSecurityEvaluationTask


@pytest.fixture
def mock_persistence():
    with patch("magda_agent.scheduler.acs_evaluation_task.ACSPersistenceV12") as MockClass:
        mock_instance = MockClass.return_value
        yield mock_instance


def test_run_evaluation_no_alerts(mock_persistence):
    """
    Test the evaluation run when no checkpoints exceed the failure rate threshold.
    """
    # Setup mock to return a low failure rate (0.05, below the 0.1 threshold)
    mock_persistence.calculate_failure_rate.return_value = 0.05

    task = ACSSecurityEvaluationTask(db_path="dummy_path.db")
    alerts = task.run_evaluation(threshold=0.1)

    assert len(alerts) == 0
    assert mock_persistence.calculate_failure_rate.call_count == 5


def test_run_evaluation_with_alerts(mock_persistence):
    """
    Test the evaluation run when some checkpoints exceed the failure rate threshold.
    """
    # Checkpoints 1 and 3 will fail above threshold, others will pass
    def side_effect(checkpoint_id):
        if checkpoint_id in (1, 3):
            return 0.20  # Exceeds 0.1
        return 0.05      # Below 0.1

    mock_persistence.calculate_failure_rate.side_effect = side_effect

    task = ACSSecurityEvaluationTask(db_path="dummy_path.db")
    alerts = task.run_evaluation(threshold=0.1)

    assert len(alerts) == 2

    checkpoint_ids = [alert["checkpoint_id"] for alert in alerts]
    assert 1 in checkpoint_ids
    assert 3 in checkpoint_ids

    for alert in alerts:
        assert alert["failure_rate"] == 0.20
        assert alert["threshold"] == 0.1
        assert "failure rate (20.00%) exceeds threshold (10.00%)" in alert["message"]


def test_run_evaluation_handles_exceptions(mock_persistence):
    """
    Test that the evaluation run handles exceptions thrown by the persistence layer gracefully.
    """
    # Checkpoint 2 raises an exception, Checkpoint 4 exceeds threshold, others are below
    def side_effect(checkpoint_id):
        if checkpoint_id == 2:
            raise Exception("Mock DB error")
        elif checkpoint_id == 4:
            return 0.15
        return 0.01

    mock_persistence.calculate_failure_rate.side_effect = side_effect

    task = ACSSecurityEvaluationTask(db_path="dummy_path.db")
    alerts = task.run_evaluation(threshold=0.1)

    # Should only return an alert for checkpoint 4
    assert len(alerts) == 1
    assert alerts[0]["checkpoint_id"] == 4
    assert alerts[0]["failure_rate"] == 0.15
