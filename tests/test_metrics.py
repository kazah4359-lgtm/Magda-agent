import pytest
from magda_agent.metacognition.tracker import QualityTracker

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
