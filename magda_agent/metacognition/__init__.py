from magda_agent.metacognition.evaluator import Evaluator
from magda_agent.metacognition.tracker import QualityTracker

# Global instance for tracking continuous improvement metrics
quality_tracker = QualityTracker()

__all__ = ["Evaluator", "QualityTracker", "quality_tracker"]
