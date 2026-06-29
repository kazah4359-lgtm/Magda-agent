import logging
import json
import hashlib
from typing import Dict, Any, Optional
from magda_agent.integration.a2a_tracing import A2ATracer

class A2ATracingV4:
    """
    Enterprise-ready tracing integration for A2A peer delegation events.
    Securely logs and extracts A2A tracing events.
    """

    @staticmethod
    def securely_log_delegation_event(target_agent_id: str, capability: str, context: Dict[str, Any], token: Optional[str] = None) -> str:
        """
        Securely logs an A2A delegation event.

        Args:
            target_agent_id: The ID of the agent being delegated to.
            capability: The capability requested.
            context: The delegation context.
            token: Optional authorization token for secure logging.

        Returns:
            str: The trace ID for the logged event.
        """
        trace_id = A2ATracer.get_or_create_trace_id()

        # Secure the context by hashing sensitive data or recording its hash
        context_str = json.dumps(context, sort_keys=True)
        context_hash = hashlib.sha256(context_str.encode('utf-8')).hexdigest()

        details = {
            "target_agent_id": target_agent_id,
            "capability": capability,
            "context_hash": context_hash,
            "secure_token_provided": bool(token)
        }

        A2ATracer.record_event("a2a_delegation_secure_v4", details)
        logging.info(f"Securely logged A2A delegation event. Trace ID: {trace_id}")

        return trace_id

    @staticmethod
    def extract_delegation_logs(trace_id: str) -> list[Dict[str, Any]]:
        """
        Extracts delegation logs in a secure, enterprise-ready format.

        Args:
            trace_id: The trace ID to extract logs for.

        Returns:
            list[Dict[str, Any]]: A list of formatted log entries.
        """
        events = A2ATracer.get_trace(trace_id)

        formatted_logs = []
        for event in events:
            if event.get("event") == "a2a_delegation_secure_v4":
                formatted_logs.append({
                    "timestamp": event.get("timestamp"),
                    "trace_id": trace_id,
                    "event_type": event.get("event"),
                    "target_agent_id": event.get("details", {}).get("target_agent_id"),
                    "capability": event.get("details", {}).get("capability"),
                    "context_hash": event.get("details", {}).get("context_hash"),
                    "is_secure": event.get("details", {}).get("secure_token_provided", False)
                })

        return formatted_logs
