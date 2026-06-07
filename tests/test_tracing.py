import pytest
from magda_agent.tracing.tracer import ThoughtChainTracer

def test_tracer_add_step():
    tracer = ThoughtChainTracer()
    assert len(tracer.get_trace()) == 0

    tracer.add_step("memory_retrieval", {"count": 3})
    trace = tracer.get_trace()
    assert len(trace) == 1
    assert trace[0]["step"] == "memory_retrieval"
    assert trace[0]["data"]["count"] == 3
    assert "timestamp" in trace[0]

def test_tracer_clear():
    tracer = ThoughtChainTracer()
    tracer.add_step("test")
    assert len(tracer.get_trace()) == 1

    tracer.clear()
    assert len(tracer.get_trace()) == 0

def test_tracer_caps_entries() -> None:
    tracer = ThoughtChainTracer(max_entries=2)
    tracer.add_step("one")
    tracer.add_step("two")
    tracer.add_step("three")
    assert [entry["step"] for entry in tracer.get_trace()] == ["one", "two"]


def test_tracer_contexts_are_independent() -> None:
    import contextvars

    tracer = ThoughtChainTracer()

    ctx1 = contextvars.copy_context()
    ctx2 = contextvars.copy_context()
    ctx1.run(tracer.add_step, "ctx1")
    ctx2.run(tracer.add_step, "ctx2")

    assert ctx1.run(tracer.get_trace)[0]["step"] == "ctx1"
    assert ctx2.run(tracer.get_trace)[0]["step"] == "ctx2"
