import pytest
from magda_agent.security.mcp_kernel_taint import mark_tainted, is_tainted, sanitize, TaintedString

def test_taint_string():
    t = mark_tainted("hello")
    assert isinstance(t, TaintedString)
    assert is_tainted(t)
    assert is_tainted("hello") is False

def test_taint_list():
    t_list = mark_tainted(["a", "b"])
    assert is_tainted(t_list)
    assert is_tainted(["a", "b"]) is False

def test_taint_dict():
    t_dict = mark_tainted({"key": "val"})
    assert is_tainted(t_dict)
    assert is_tainted({"key": "val"}) is False

def test_sanitize():
    t_dict = mark_tainted({"key": ["val", "x"]})
    clean = sanitize(t_dict)
    assert is_tainted(clean) is False
    assert clean == {"key": ["val", "x"]}
