import logging
from typing import Dict, Any, List

from magda_agent.safety.acs_persistence_v12 import ACSPersistenceV12

class ACSSecurityEvaluationTask:
    """
    Periodic task to evaluate the ACS checkpoint persistence database
    for policy violation trends.
    """

    def __init__(self, db_path: str = "acs_persistence_v12.db") -> None:
        """
        Initializes the evaluation task.

        Args:
            db_path: Path to the SQLite database file for ACS persistence.
        """
        self.logger = logging.getLogger(__name__)
        self.persistence = ACSPersistenceV12(db_path=db_path)

    def run_evaluation(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Evaluates the failure rates for ACS checkpoints 1-5.

        Args:
            threshold: The failure rate threshold above which an alert is generated (0.0 to 1.0).

        Returns:
            A list of dictionary alerts for checkpoints exceeding the failure threshold.
        """
        alerts = []
        for checkpoint_id in range(1, 6):
            try:
                failure_rate = self.persistence.calculate_failure_rate(checkpoint_id=checkpoint_id)
                if failure_rate > threshold:
                    alerts.append({
                        "checkpoint_id": checkpoint_id,
                        "failure_rate": failure_rate,
                        "threshold": threshold,
                        "message": f"Checkpoint {checkpoint_id} failure rate ({failure_rate:.2%}) exceeds threshold ({threshold:.2%})."
                    })
                    self.logger.warning(f"Alert generated for Checkpoint {checkpoint_id}: Failure rate {failure_rate:.2%} > {threshold:.2%}")
            except Exception as e:
                self.logger.error(f"Error evaluating checkpoint {checkpoint_id}: {e}")
        return alerts
