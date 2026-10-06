"""
Agent Teams Distributed Tracing V9.

Inspired by Claude Agent SDK Agent Teams and enterprise OpenTelemetry tracing trends:
Implements distributed tracing that propagates OpenTelemetry context across sub-agents
running in isolated Git worktrees, allowing full visibility into parallel tasks.
"""

import time
import uuid
import logging
from contextlib import contextmanager, asynccontextmanager
from typing import Dict, Any, List, Optional, Tuple, Generator, AsyncGenerator

logger = logging.getLogger(__name__)


def generate_trace_id() -> str:
    """Generate a 32-character hex trace ID conforming to W3C Trace Context spec."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate a 16-character hex span ID conforming to W3C Trace Context spec."""
    return uuid.uuid4().hex[:16]


class TraceSpanV9:
    """
    Represents an OpenTelemetry-compatible span within a distributed agent trace.
    """

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        operation_name: str,
        agent_id: str,
        parent_span_id: Optional[str] = None,
        subagent_id: Optional[str] = None,
        worktree_path: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.trace_id = trace_id
        self.span_id = span_id
        self.operation_name = operation_name
        self.agent_id = agent_id
        self.parent_span_id = parent_span_id
        self.subagent_id = subagent_id
        self.worktree_path = worktree_path
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.status: str = "active"
        self.error_message: Optional[str] = None
        self.attributes: Dict[str, Any] = attributes or {}

    def finish(self, status: str = "completed", error_message: Optional[str] = None) -> None:
        """Mark the span as finished with end timestamp and status."""
        self.end_time = time.time()
        self.status = status
        if error_message:
            self.error_message = error_message
            self.attributes["error"] = error_message

    def to_traceparent(self) -> str:
        """
        Format span context as a W3C traceparent string:
        version(00)-trace_id(32 hex)-span_id(16 hex)-trace_flags(01)
        """
        return f"00-{self.trace_id}-{self.span_id}-01"

    def to_dict(self) -> Dict[str, Any]:
        """Convert span details to a dictionary representation."""
        duration = (self.end_time - self.start_time) if self.end_time else None
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "agent_id": self.agent_id,
            "subagent_id": self.subagent_id,
            "worktree_path": self.worktree_path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": duration,
            "status": self.status,
            "error_message": self.error_message,
            "attributes": dict(self.attributes),
            "traceparent": self.to_traceparent(),
        }


class AgentTeamsTracingManagerV9:
    """
    Manages distributed tracing across Agent Teams sub-agents operating in isolated Git worktrees.
    Tracks root traces, child spans, propagates context via environment variables and traceparent headers,
    and aggregates trace trees for monitoring and debugging.
    """

    def __init__(self, service_name: str = "magda_agent_teams") -> None:
        self.service_name = service_name
        self.spans: Dict[str, TraceSpanV9] = {}  # span_id -> TraceSpanV9
        self.traces: Dict[str, List[str]] = {}  # trace_id -> List[span_id]

    def start_trace(
        self,
        agent_id: str,
        operation_name: str = "orchestrate_team",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> TraceSpanV9:
        """
        Starts a root trace and returns the root span.
        """
        trace_id = generate_trace_id()
        span_id = generate_span_id()
        span = TraceSpanV9(
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation_name,
            agent_id=agent_id,
            parent_span_id=None,
            attributes=attributes,
        )
        self.spans[span_id] = span
        self.traces.setdefault(trace_id, []).append(span_id)
        logger.debug(f"Started trace {trace_id} with root span {span_id} for agent {agent_id}")
        return span

    def spawn_subagent_trace(
        self,
        parent_span: TraceSpanV9,
        subagent_id: str,
        worktree_path: Optional[str] = None,
        operation_name: str = "subagent_task",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Tuple[TraceSpanV9, Dict[str, str]]:
        """
        Spawns a child span for a sub-agent operating in an isolated Git worktree,
        and generates isolated environment variables / trace context headers for propagation.

        Args:
            parent_span (TraceSpanV9): Parent agent span.
            subagent_id (str): Identifier for spawned sub-agent.
            worktree_path (Optional[str]): Isolated Git worktree path.
            operation_name (str): Sub-agent task operation name.
            attributes (Optional[Dict[str, Any]]): Additional attributes.

        Returns:
            Tuple[TraceSpanV9, Dict[str, str]]: The child span and an environment dictionary for propagation.
        """
        child_span_id = generate_span_id()
        attr = dict(attributes or {})
        if worktree_path:
            attr["worktree_path"] = worktree_path

        child_span = TraceSpanV9(
            trace_id=parent_span.trace_id,
            span_id=child_span_id,
            operation_name=operation_name,
            agent_id=subagent_id,
            parent_span_id=parent_span.span_id,
            subagent_id=subagent_id,
            worktree_path=worktree_path,
            attributes=attr,
        )

        self.spans[child_span_id] = child_span
        self.traces.setdefault(parent_span.trace_id, []).append(child_span_id)

        # Build OpenTelemetry propagation environment variables
        env_vars = {
            "OTEL_TRACEPARENT": child_span.to_traceparent(),
            "TRACEPARENT": child_span.to_traceparent(),
            "MAGDA_TRACE_ID": parent_span.trace_id,
            "MAGDA_SPAN_ID": child_span_id,
            "MAGDA_PARENT_SPAN_ID": parent_span.span_id,
            "MAGDA_SUBAGENT_ID": subagent_id,
        }
        if worktree_path:
            env_vars["MAGDA_WORKTREE_PATH"] = worktree_path

        logger.debug(
            f"Spawned subagent trace for {subagent_id}: trace_id={parent_span.trace_id}, "
            f"child_span_id={child_span_id}, parent_span_id={parent_span.span_id}"
        )
        return child_span, env_vars

    def extract_trace_context(self, env_vars: Dict[str, str]) -> Dict[str, Any]:
        """
        Extracts tracing context from sub-agent environment variables or headers.

        Args:
            env_vars (Dict[str, str]): Environment variables or headers dict.

        Returns:
            Dict[str, Any]: Extracted trace context dictionary.
        """
        traceparent = env_vars.get("OTEL_TRACEPARENT") or env_vars.get("TRACEPARENT")
        trace_id = env_vars.get("MAGDA_TRACE_ID")
        span_id = env_vars.get("MAGDA_SPAN_ID")
        parent_span_id = env_vars.get("MAGDA_PARENT_SPAN_ID")
        subagent_id = env_vars.get("MAGDA_SUBAGENT_ID")
        worktree_path = env_vars.get("MAGDA_WORKTREE_PATH")

        if traceparent and "-" in traceparent:
            parts = traceparent.split("-")
            if len(parts) >= 4:
                trace_id = trace_id or parts[1]
                span_id = span_id or parts[2]

        return {
            "traceparent": traceparent,
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "subagent_id": subagent_id,
            "worktree_path": worktree_path,
        }

    def record_subagent_span(
        self,
        trace_id: str,
        span_id: str,
        agent_id: str,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        status: str = "completed",
        error_message: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> TraceSpanV9:
        """
        Records or finishes a sub-agent span directly using extracted tracing context.
        """
        if span_id in self.spans:
            span = self.spans[span_id]
            span.finish(status=status, error_message=error_message)
            if attributes:
                span.attributes.update(attributes)
            return span

        span = TraceSpanV9(
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation_name,
            agent_id=agent_id,
            parent_span_id=parent_span_id,
            subagent_id=agent_id,
            attributes=attributes,
        )
        span.finish(status=status, error_message=error_message)
        self.spans[span_id] = span
        self.traces.setdefault(trace_id, []).append(span_id)
        return span

    def end_span(
        self, span_id: str, status: str = "completed", error_message: Optional[str] = None
    ) -> Optional[TraceSpanV9]:
        """Mark a span as ended."""
        span = self.spans.get(span_id)
        if span:
            span.finish(status=status, error_message=error_message)
        return span

    def get_trace_spans(self, trace_id: str) -> List[TraceSpanV9]:
        """Retrieve all spans belonging to a trace_id."""
        span_ids = self.traces.get(trace_id, [])
        return [self.spans[sid] for sid in span_ids if sid in self.spans]

    def export_trace_tree(self, trace_id: str) -> Dict[str, Any]:
        """
        Exports a trace hierarchy tree containing parent-child span relationships.
        """
        spans = self.get_trace_spans(trace_id)
        if not spans:
            return {"trace_id": trace_id, "spans_count": 0, "tree": {}}

        root_spans = [s for s in spans if not s.parent_span_id or s.parent_span_id not in self.spans]
        children_map: Dict[str, List[TraceSpanV9]] = {}
        for s in spans:
            if s.parent_span_id:
                children_map.setdefault(s.parent_span_id, []).append(s)

        def _build_node(span: TraceSpanV9) -> Dict[str, Any]:
            node = span.to_dict()
            children = children_map.get(span.span_id, [])
            node["children"] = [_build_node(c) for c in children]
            return node

        tree_nodes = [_build_node(r) for r in root_spans]

        return {
            "trace_id": trace_id,
            "spans_count": len(spans),
            "roots": tree_nodes,
        }

    def clear(self) -> None:
        """Clear all stored spans and traces."""
        self.spans.clear()
        self.traces.clear()


@contextmanager
def trace_subagent_span(
    manager: AgentTeamsTracingManagerV9,
    parent_span: TraceSpanV9,
    subagent_id: str,
    worktree_path: Optional[str] = None,
    operation_name: str = "subagent_exec",
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[Tuple[TraceSpanV9, Dict[str, str]], None, None]:
    """
    Synchronous context manager that automatically manages child sub-agent tracing context.
    """
    child_span, env_vars = manager.spawn_subagent_trace(
        parent_span=parent_span,
        subagent_id=subagent_id,
        worktree_path=worktree_path,
        operation_name=operation_name,
        attributes=attributes,
    )
    try:
        yield child_span, env_vars
        child_span.finish(status="completed")
    except Exception as exc:
        child_span.finish(status="error", error_message=str(exc))
        raise


@asynccontextmanager
async def trace_subagent_span_async(
    manager: AgentTeamsTracingManagerV9,
    parent_span: TraceSpanV9,
    subagent_id: str,
    worktree_path: Optional[str] = None,
    operation_name: str = "subagent_exec",
    attributes: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Tuple[TraceSpanV9, Dict[str, str]], None]:
    """
    Asynchronous context manager that automatically manages child sub-agent tracing context.
    """
    child_span, env_vars = manager.spawn_subagent_trace(
        parent_span=parent_span,
        subagent_id=subagent_id,
        worktree_path=worktree_path,
        operation_name=operation_name,
        attributes=attributes,
    )
    try:
        yield child_span, env_vars
        child_span.finish(status="completed")
    except Exception as exc:
        child_span.finish(status="error", error_message=str(exc))
        raise
