import time
from typing import Dict, Any, Optional
from magda_agent.safety.acs_guard import ACSGuard, SecurityViolationError
from magda_agent.safety.audit_trail import AuditTrail

class ACSAuditLogger:
    """
    Audit logging mechanism that records all tools evaluated and intercepted
    by the ACSGuard policy layer.
    """

    def __init__(self, acs_guard: ACSGuard, audit_trail: Optional[AuditTrail] = None) -> None:
        """
        Initializes the ACSAuditLogger.

        Args:
            acs_guard: The ACSGuard instance to wrap.
            audit_trail: The AuditTrail instance to log to.
        """
        self.acs_guard = acs_guard
        self.audit_trail = audit_trail or AuditTrail()

    def evaluate_and_intercept(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wraps ACSGuard.intercept_action to log the result to AuditTrail.

        Args:
            workflow_data: The data for the action to validate.

        Returns:
            The unmodified workflow data if it passed validation.

        Raises:
            SecurityViolationError: If validation fails at any checkpoint.
        """
        start_time = time.time()
        tool_name = workflow_data.get("tool", "unknown")

        try:
            result = self.acs_guard.intercept_action(workflow_data)
            duration = time.time() - start_time
            self.audit_trail.log_call(
                tool_name=tool_name,
                kwargs=workflow_data.get("kwargs", {}),
                why="ACSGuard evaluation passed",
                result="allowed",
                duration=duration
            )
            return result
        except SecurityViolationError as e:
            duration = time.time() - start_time
            self.audit_trail.log_call(
                tool_name=tool_name,
                kwargs=workflow_data.get("kwargs", {}),
                why=f"ACSGuard evaluation failed: {str(e)}",
                result="blocked",
                duration=duration
            )
            raise
