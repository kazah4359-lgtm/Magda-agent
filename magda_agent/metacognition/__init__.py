from magda_agent.metacognition.evaluator import Evaluator
from magda_agent.metacognition.confidence import ConfidenceCalibrator
from magda_agent.metacognition.tracker import QualityTracker
from magda_agent.metacognition.failure_patterns import FailurePatternTracker

# Global instance for tracking continuous improvement metrics
quality_tracker = QualityTracker()

__all__ = ["Evaluator", "QualityTracker", "quality_tracker", "FailurePatternTracker", "ConfidenceCalibrator"]
