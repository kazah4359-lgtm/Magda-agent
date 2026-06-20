from magda_agent.metacognition.metrics import LongitudinalMetrics
import pytest
import sys
import io
import tempfile
import os
from unittest.mock import patch
from magda_agent.metacognition.tracker import QualityTracker
from magda_agent.metacognition.metrics_cli import main

@pytest.fixture
def quality_tracker():
    """Provides a QualityTracker instance using an ephemeral SQLite database for testing."""
    return QualityTracker(db_path=":memory:")

def test_log_and_get_metrics(quality_tracker):
    """Test that metrics can be logged and correctly retrieved."""
    # Log some test metrics
    quality_tracker.log_metric("test_pass_rate", 95.5, {"commit": "abc1234"})
    quality_tracker.log_metric("test_pass_rate", 98.0, {"commit": "def5678"})
    quality_tracker.log_metric("pr_size", 120.0, {"commit": "abc1234"})

    # Retrieve metrics for test_pass_rate
    pass_rate_metrics = quality_tracker.get_metrics("test_pass_rate")

    assert len(pass_rate_metrics) == 2

    # Values should be present in the returned metadata
    values = [m["value"] for m in pass_rate_metrics]
    assert 95.5 in values
    assert 98.0 in values

    # Retrieve metrics for pr_size
    pr_size_metrics = quality_tracker.get_metrics("pr_size")
    assert len(pr_size_metrics) == 1
    assert pr_size_metrics[0]["value"] == 120.0

def test_calculate_average(quality_tracker):
    """Test the calculation of average metric values."""
    # Log multiple values for a metric
    quality_tracker.log_metric("test_pass_rate", 90.0)
    quality_tracker.log_metric("test_pass_rate", 100.0)
    quality_tracker.log_metric("test_pass_rate", 80.0)

    # Calculate average
    avg = quality_tracker.calculate_average("test_pass_rate")
    assert avg == 90.0

def test_calculate_average_empty(quality_tracker):
    """Test calculating average when no metrics exist."""
    avg = quality_tracker.calculate_average("non_existent_metric")
    assert avg is None

def test_get_metrics_limit(quality_tracker):
    """Test that get_metrics respects the limit parameter."""
    for i in range(15):
        quality_tracker.log_metric("pr_size", float(i))

    metrics = quality_tracker.get_metrics("pr_size", limit=5)
    assert len(metrics) <= 5






def test_metrics_cli_list(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the list command from the CLI."""
    # Setup test DB (we need to pass the same in-memory DB or temporary file)
    # The CLI creates a new QualityTracker, so we'll use a temporary file instead of :memory:
    # to avoid the DB disappearing between tracker instances.
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        db_path = tmp.name

    try:
        # Create a tracker and log some entries
        tracker = QualityTracker(db_path=db_path)
        tracker.log_metric("test_metric", 10.5, {"info": "first"})
        tracker.log_metric("test_metric", 20.0, {"info": "second"})

        # Mock sys.argv to simulate running the CLI
        test_args = ["metrics_cli.py", "list", "test_metric", "--db", db_path]
        with patch.object(sys, 'argv', test_args):
            main()

        # Capture output
        captured = capsys.readouterr()

        # Verify the output contains the metrics
        assert "Recent metrics for test_metric:" in captured.out
        assert "10.5" in captured.out
        assert "20.0" in captured.out
        assert "first" in captured.out
        assert "second" in captured.out
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

def test_metrics_cli_list_empty(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the list command from the CLI when no metrics exist."""
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        db_path = tmp.name

    try:
        test_args = ["metrics_cli.py", "list", "non_existent_metric", "--db", db_path]
        with patch.object(sys, 'argv', test_args):
            main()

        captured = capsys.readouterr()
        assert "No entries found for non_existent_metric" in captured.out
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)



@pytest.fixture
def longitudinal_metrics():
    """Provides a LongitudinalMetrics instance using an ephemeral SQLite database for testing."""
    return LongitudinalMetrics(db_path=":memory:")

def test_longitudinal_metrics_log_and_get(longitudinal_metrics):
    """Test logging and retrieving test success rates."""
    longitudinal_metrics.log_test_success_rate("core_suite", 100, 95)
    longitudinal_metrics.log_test_success_rate("core_suite", 50, 50)

    rates = longitudinal_metrics.get_success_rates("core_suite")
    assert len(rates) == 2

    # The rates should have both 100.0 and 95.0
    success_rates = [r['success_rate'] for r in rates]
    assert 100.0 in success_rates
    assert 95.0 in success_rates

def test_longitudinal_metrics_log_zero_total(longitudinal_metrics, caplog):
    """Test logging test success rate with zero total tests."""
    longitudinal_metrics.log_test_success_rate("empty_suite", 0, 0)

    rates = longitudinal_metrics.get_success_rates("empty_suite")
    assert len(rates) == 0
    assert "Cannot log success rate with total_tests <= 0" in caplog.text

def test_longitudinal_metrics_limit(longitudinal_metrics):
    """Test that retrieval respects the limit parameter."""
    for i in range(15):
        longitudinal_metrics.log_test_success_rate("loop_suite", 10, i % 10)

    rates = longitudinal_metrics.get_success_rates("loop_suite", limit=5)
    assert len(rates) == 5
