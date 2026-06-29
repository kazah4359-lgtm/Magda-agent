import pytest
import hashlib
import json
from unittest.mock import patch, MagicMock
from magda_agent.integration.a2a_tracing_v4 import A2ATracingV4
from magda_agent.integration.a2a_tracing import A2ATracer

@pytest.fixture(autouse=True)
def reset_tracer() -> None:
    """
    Fixture to reset tracer state before each test.
    """
    # Clear tracing data before each test
    A2ATracer.clear_registry()
    A2ATracer.set_trace_id(None)
    yield
    A2ATracer.clear_registry()

def test_securely_log_delegation_event() -> None:
    """
    Tests that securely logging a delegation event correctly generates a trace
    and the extracted logs match the expected format with hashing.
    """
    target_id = "agent-123"
    capability = "code_execution"
    context = {"task": "run script"}
    token = "secure-token"

    # Pre-calculate expected hash
    context_str = json.dumps(context, sort_keys=True)
    expected_hash = hashlib.sha256(context_str.encode('utf-8')).hexdigest()

    trace_id = A2ATracingV4.securely_log_delegation_event(target_id, capability, context, token)

    assert trace_id is not None
    assert trace_id == A2ATracer.get_current_trace_id()

    logs = A2ATracingV4.extract_delegation_logs(trace_id)
    assert len(logs) == 1

    log = logs[0]
    assert log["trace_id"] == trace_id
    assert log["event_type"] == "a2a_delegation_secure_v4"
    assert log["target_agent_id"] == target_id
    assert log["capability"] == capability
    assert log["context_hash"] == expected_hash
    assert log["is_secure"] is True

def test_securely_log_delegation_event_no_token() -> None:
    """
    Tests securely logging a delegation event when no token is provided.
    """
    target_id = "agent-456"
    capability = "search"
    context = {"query": "test query"}

    trace_id = A2ATracingV4.securely_log_delegation_event(target_id, capability, context)

    logs = A2ATracingV4.extract_delegation_logs(trace_id)
    assert len(logs) == 1

    log = logs[0]
    assert log["is_secure"] is False

def test_extract_delegation_logs_empty() -> None:
    """
    Tests extraction behavior when passing a non-existent trace ID.
    """
    logs = A2ATracingV4.extract_delegation_logs("non-existent-trace")
    assert len(logs) == 0

def test_extract_delegation_logs_filters_events() -> None:
    """
    Tests that extraction correctly filters only the secure v4 delegation events.
    """
    # Record a normal event
    trace_id = A2ATracer.get_or_create_trace_id()
    A2ATracer.record_event("some_other_event", {"key": "value"})

    # Record a secure delegation event
    A2ATracingV4.securely_log_delegation_event("agent-999", "test", {})

    # It should only extract the secure delegation event
    logs = A2ATracingV4.extract_delegation_logs(trace_id)
    assert len(logs) == 1
    assert logs[0]["event_type"] == "a2a_delegation_secure_v4"
