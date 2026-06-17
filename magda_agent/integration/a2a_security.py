import uuid
import logging
from typing import Dict, Any
from magda_agent.integration.a2a_tracing import A2ATracer

class A2ASecurityContext:
    """
    Provides enterprise-ready async auth and tracing for A2A communications.
    """
    def __init__(self) -> None:
        """Initializes the security context."""
        self._active_tokens = set()

    def generate_token(self) -> str:
        """
        Generates a new authentication token for A2A requests.

        Returns:
            str: The generated token.
        """
        token = f"a2a_{uuid.uuid4().hex}"
        self._active_tokens.add(token)
        return token

    def validate_token(self, token: str) -> bool:
        """
        Validates an A2A authentication token.

        Args:
            token: The token to validate.

        Returns:
            bool: True if the token is valid, False otherwise.
        """
        return token in self._active_tokens

    def trace_action(self, action: str, details: Dict[str, Any]) -> str:
        """
        Logs an action in the audit trail and returns a trace ID.
        Uses A2ATracer to ensure distributed tracing continuity.

        Args:
            action: The action name.
            details: Action details for the audit trail.

        Returns:
            str: The current or generated trace ID.
        """
        trace_id = A2ATracer.get_or_create_trace_id()
        logging.info(f"[AUDIT TRAIL] TraceID: {trace_id} | Action: {action} | Details: {details}")
        return trace_id
