from magda_agent.metacognition.evaluator import Evaluator
from magda_agent.metacognition.assert_evaluator import AssertEvaluator
from magda_agent.metacognition.confidence import ConfidenceCalibrator
from magda_agent.metacognition.tracker import QualityTracker
from magda_agent.metacognition.failure_patterns import FailurePatternTracker

from magda_agent.metacognition.metrics import LongitudinalMetrics

# Global instance for tracking continuous improvement metrics
quality_tracker = QualityTracker()
longitudinal_metrics = LongitudinalMetrics()

__all__ = ["AssertEvaluator", "Evaluator", "QualityTracker", "quality_tracker", "FailurePatternTracker", "ConfidenceCalibrator", "LongitudinalMetrics", "longitudinal_metrics"]
