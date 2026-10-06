import pytest
import asyncio
from magda_agent.architecture.agent_teams_tracing_v9 import (
    AgentTeamsTracingManagerV9,
    TraceSpanV9,
    generate_trace_id,
    generate_span_id,
    trace_subagent_span,
    trace_subagent_span_async,
)


def test_generate_ids():
    trace_id = generate_trace_id()
    span_id = generate_span_id()
    assert isinstance(trace_id, str)
    assert len(trace_id) == 32
    assert isinstance(span_id, str)
    assert len(span_id) == 16


def test_trace_span_lifecycle():
    span = TraceSpanV9(
        trace_id="12345678901234567890123456789012",
        span_id="1234567890123456",
        operation_name="test_op",
        agent_id="agent_1",
    )
    assert span.status == "active"
    assert span.end_time is None
    assert span.to_traceparent() == "00-12345678901234567890123456789012-1234567890123456-01"

    span.finish(status="completed")
    assert span.status == "completed"
    assert span.end_time is not None
    span_dict = span.to_dict()
    assert span_dict["trace_id"] == "12345678901234567890123456789012"
    assert span_dict["duration"] >= 0.0


def test_tracing_manager_start_and_spawn():
    manager = AgentTeamsTracingManagerV9()
    root_span = manager.start_trace(agent_id="orchestrator", operation_name="parallel_planning")

    assert root_span.trace_id is not None
    assert root_span.parent_span_id is None

    worktree_path = "/tmp/worktrees/agent_coder_1"
    child_span, env_vars = manager.spawn_subagent_trace(
        parent_span=root_span,
        subagent_id="coder_subagent_1",
        worktree_path=worktree_path,
        operation_name="code_feature",
    )

    # 1. Tracing context propagates to spawned sub-agents
    assert child_span.trace_id == root_span.trace_id
    assert child_span.parent_span_id == root_span.span_id
    assert child_span.subagent_id == "coder_subagent_1"

    # 2. Environment variables propagated
    assert env_vars["MAGDA_TRACE_ID"] == root_span.trace_id
    assert env_vars["MAGDA_SPAN_ID"] == child_span.span_id
    assert env_vars["MAGDA_PARENT_SPAN_ID"] == root_span.span_id
    assert env_vars["MAGDA_WORKTREE_PATH"] == worktree_path
    assert env_vars["OTEL_TRACEPARENT"] == child_span.to_traceparent()


def test_extract_trace_context():
    manager = AgentTeamsTracingManagerV9()
    env_vars = {
        "OTEL_TRACEPARENT": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "MAGDA_TRACE_ID": "4bf92f3577b34da6a3ce929d0e0e4736",
        "MAGDA_SPAN_ID": "00f067aa0ba902b7",
        "MAGDA_PARENT_SPAN_ID": "0000000000000001",
        "MAGDA_SUBAGENT_ID": "worker_1",
        "MAGDA_WORKTREE_PATH": "/tmp/worktree_1",
    }

    ctx = manager.extract_trace_context(env_vars)
    assert ctx["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert ctx["span_id"] == "00f067aa0ba902b7"
    assert ctx["parent_span_id"] == "0000000000000001"
    assert ctx["subagent_id"] == "worker_1"
    assert ctx["worktree_path"] == "/tmp/worktree_1"


def test_sync_trace_context_manager():
    manager = AgentTeamsTracingManagerV9()
    root_span = manager.start_trace(agent_id="orchestrator")

    with trace_subagent_span(
        manager=manager,
        parent_span=root_span,
        subagent_id="sub_1",
        worktree_path="/tmp/sub_1_worktree",
    ) as (child_span, env_vars):
        assert child_span.trace_id == root_span.trace_id
        assert env_vars["MAGDA_TRACE_ID"] == root_span.trace_id
        assert child_span.status == "active"

    assert child_span.status == "completed"
    assert child_span.end_time is not None


def test_sync_trace_context_manager_exception():
    manager = AgentTeamsTracingManagerV9()
    root_span = manager.start_trace(agent_id="orchestrator")

    with pytest.raises(ValueError, match="Task failed"):
        with trace_subagent_span(
            manager=manager,
            parent_span=root_span,
            subagent_id="sub_error",
        ) as (child_span, _):
            raise ValueError("Task failed")

    assert child_span.status == "error"
    assert "Task failed" in child_span.error_message


def test_async_trace_context_manager():
    async def _run():
        manager = AgentTeamsTracingManagerV9()
        root_span = manager.start_trace(agent_id="orchestrator")

        async with trace_subagent_span_async(
            manager=manager,
            parent_span=root_span,
            subagent_id="async_sub_1",
            worktree_path="/tmp/async_worktree",
        ) as (child_span, env_vars):
            await asyncio.sleep(0.01)
            assert child_span.trace_id == root_span.trace_id
            assert env_vars["MAGDA_SUBAGENT_ID"] == "async_sub_1"

        assert child_span.status == "completed"

    asyncio.run(_run())


def test_export_trace_tree():
    manager = AgentTeamsTracingManagerV9()
    root_span = manager.start_trace(agent_id="parent_agent", operation_name="multi_task_plan")

    child1, env1 = manager.spawn_subagent_trace(
        parent_span=root_span,
        subagent_id="subagent_1",
        worktree_path="/tmp/wt1",
        operation_name="task_a",
    )

    child2, env2 = manager.spawn_subagent_trace(
        parent_span=root_span,
        subagent_id="subagent_2",
        worktree_path="/tmp/wt2",
        operation_name="task_b",
    )

    # Sub-child span off subagent_1
    sub_child, _ = manager.spawn_subagent_trace(
        parent_span=child1,
        subagent_id="subagent_1_1",
        worktree_path="/tmp/wt1_1",
        operation_name="sub_task_a1",
    )

    tree = manager.export_trace_tree(root_span.trace_id)
    assert tree["trace_id"] == root_span.trace_id
    assert tree["spans_count"] == 4
    assert len(tree["roots"]) == 1

    root_node = tree["roots"][0]
    assert root_node["agent_id"] == "parent_agent"
    assert len(root_node["children"]) == 2

    child1_node = next(c for c in root_node["children"] if c["subagent_id"] == "subagent_1")
    assert len(child1_node["children"]) == 1
    assert child1_node["children"][0]["subagent_id"] == "subagent_1_1"


def test_record_subagent_span_and_clear():
    manager = AgentTeamsTracingManagerV9()
    trace_id = generate_trace_id()
    span_id = generate_span_id()

    span = manager.record_subagent_span(
        trace_id=trace_id,
        span_id=span_id,
        agent_id="external_subagent",
        operation_name="run_script",
        status="completed",
        attributes={"exit_code": 0},
    )

    assert span.trace_id == trace_id
    assert span.span_id == span_id
    assert span.attributes["exit_code"] == 0

    spans = manager.get_trace_spans(trace_id)
    assert len(spans) == 1

    manager.clear()
    assert len(manager.get_trace_spans(trace_id)) == 0
