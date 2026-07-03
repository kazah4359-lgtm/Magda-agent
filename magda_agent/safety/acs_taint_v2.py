"""
ACS Taint Tracking Checkpoint v2 module.
Implements taint tracking for the output sanitization checkpoint.
"""

from typing import Dict, Any, Tuple
from magda_agent.safety.acs_guard import ACSCheckpoint
from magda_agent.safety.taint import is_tainted


class OutputTaintTrackingCheckpointV2(ACSCheckpoint):
    """
    Checkpoint: Output Taint Tracking.
    Verifies that the output in the workflow data is not tainted.
    """

    def validate(self, workflow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates the workflow data against this checkpoint.

        Args:
            workflow_data: The workflow context data.

        Returns:
            A tuple (passed, reason).
        """
        if "output" not in workflow_data:
            return True, "Output taint tracking passed: no output to validate."

        output = workflow_data["output"]

        if is_tainted(output):
            return False, "Output taint tracking failed: tainted data detected in output."

        return True, "Output taint tracking passed."
