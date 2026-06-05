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
