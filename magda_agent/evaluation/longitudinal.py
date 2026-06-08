from magda_agent.metacognition.tracker import QualityTracker
from typing import Dict, Any, List

class LongitudinalEvaluator:
    """Evaluates agent performance over multiple iterations."""
    def __init__(self, db_path: str = "./metrics_db.sqlite3"):
        self.tracker = QualityTracker(db_path=db_path)

    def evaluate_trend(self, metric_name: str, limit: int = 10) -> Dict[str, Any]:
        """Calculates trend and average over multiple entries."""
        metrics = self.tracker.get_metrics(metric_name, limit)
        if not metrics:
            return {"metric_name": metric_name, "trend": "unknown", "average": 0.0, "data_points": 0}

        # metrics are ordered by timestamp DESC
        values = [m["value"] for m in metrics]
        avg = sum(values) / len(values)

        trend = "stable"
        if len(values) > 1:
            # values[0] is the newest, values[-1] is the oldest
            if values[0] > values[-1]:
                trend = "improving"
            elif values[0] < values[-1]:
                trend = "declining"

        return {
            "metric_name": metric_name,
            "trend": trend,
            "average": avg,
            "data_points": len(values)
        }
