import logging
from typing import Dict, Any, Tuple, Optional

from magda_agent.safety.acs_checkpoints import ACSCheckpoints, CheckpointStage

class AdaptiveGuardrail:
    """
    Risk-based adaptive guardrails that dynamically enable/disable ACS checkpoints based on real-time risk scoring.
    """

    def __init__(self, acs_checkpoints: Optional[ACSCheckpoints] = None) -> None:
        """
        Initializes the AdaptiveGuardrail.

        Args:
            acs_checkpoints: An instance of ACSCheckpoints to run the underlying validation.
        """
        self.logger = logging.getLogger(__name__)
        self.acs_checkpoints = acs_checkpoints or ACSCheckpoints()

    def evaluate(self, workflow_data: Dict[str, Any], risk_score: str) -> Tuple[bool, str]:
        """
        Dynamically run validation stages based on the risk score.

        Args:
            workflow_data: The data payload to evaluate.
            risk_score: A string representing risk ("low", "medium", "high", "critical").

        Returns:
            A tuple of (is_passed, message).
        """
        # Determine stages to run based on risk score
        stages_to_run = []
        if risk_score == "low":
            stages_to_run = [CheckpointStage.INPUT]
        elif risk_score == "medium":
            stages_to_run = [CheckpointStage.INPUT, CheckpointStage.EXECUTION]
        elif risk_score in ["high", "critical"]:
            stages_to_run = [CheckpointStage.INPUT, CheckpointStage.EXECUTION, CheckpointStage.OUTPUT]
        else:
            # Default to running everything if risk is unknown
            self.logger.warning(f"Unknown risk score '{risk_score}'. Defaulting to all checkpoints.")
            stages_to_run = [CheckpointStage.INPUT, CheckpointStage.EXECUTION, CheckpointStage.OUTPUT]

        # Execute selected stages sequentially
        for stage in stages_to_run:
            passed, reason = self.acs_checkpoints._run_stage(stage, workflow_data)
            if not passed:
                self.logger.warning(f"Adaptive validation failed at {stage.name}: {reason}")
                return False, reason

        return True, f"Adaptive guardrails passed for risk level: {risk_score}."
