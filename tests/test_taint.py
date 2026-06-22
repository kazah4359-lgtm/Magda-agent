"""Tests for the MCPKernel Taint Tracking Sandbox."""
import pytest
from typing import Dict, Any, List
from magda_agent.safety.taint import MCPKernel, PolicyViolationError, mark_tainted, is_tainted, sanitize

def sample_sensitive_tool(data: Any) -> str:
    """A sample sensitive tool that returns processed data."""
    return f"Processed {data}"

def sample_safe_tool(data: Any) -> str:
    """A sample safe tool that returns data."""
    return f"Safe {data}"

def test_mcp_kernel_taint_tracking() -> None:
    """Test that the MCPKernel properly tracks and rejects tainted inputs."""
    kernel = MCPKernel()

    # Create inputs
    safe_data = {"user": "alice"}
    tainted_string = kernel.tracker.taint("DROP TABLE users;")
    tainted_data = {"payload": tainted_string}

    assert kernel.tracker.is_tainted(tainted_string) == True
    assert kernel.tracker.is_tainted(safe_data) == False
    assert kernel.tracker.is_tainted(tainted_data) == True

    # Safe tool should accept untainted
    res = kernel.execute_tool(sample_safe_tool, {"data": safe_data}, is_sensitive=False)
    assert "Safe" in res
    assert not kernel.tracker.is_tainted(res)

    # Safe tool should accept tainted (not sensitive)
    res = kernel.execute_tool(sample_safe_tool, {"data": tainted_data}, is_sensitive=False)
    assert "Safe" in res
    assert kernel.tracker.is_tainted(res) # Taint propagation

    # Sensitive tool should reject tainted
    with pytest.raises(PolicyViolationError) as exc_info:
        kernel.execute_tool(sample_sensitive_tool, {"data": tainted_string}, is_sensitive=True)
    assert "Tainted input to sensitive tool call" in str(exc_info.value)

    # Sensitive tool should reject nested tainted data
    with pytest.raises(PolicyViolationError) as exc_info:
        kernel.execute_tool(sample_sensitive_tool, {"data": tainted_data}, is_sensitive=True)
    assert "Tainted input to sensitive tool call" in str(exc_info.value)

    # Sensitive tool should accept safe data
    res = kernel.execute_tool(sample_sensitive_tool, {"data": safe_data}, is_sensitive=True)
    assert "Processed" in res
    assert not kernel.tracker.is_tainted(res)

def test_recursive_taint_tracking():
    """Test recursive marking and checking of tainted data."""
    data = {
        "users": ["alice", "bob"],
        "metadata": {
            "query": "SELECT * FROM users"
        }
    }

    # Mark specific part tainted
    data["metadata"]["query"] = mark_tainted(data["metadata"]["query"])
    assert is_tainted(data) == True
    assert is_tainted(data["metadata"]) == True
    assert is_tainted(data["users"]) == False

    # mark_tainted on entire structure
    tainted_struct = mark_tainted({
        "a": [1, "unsafe"],
        "b": {"c": "dangerous"}
    })
    assert is_tainted(tainted_struct) == True
    assert is_tainted(tainted_struct["a"][1]) == True
    assert is_tainted(tainted_struct["b"]["c"]) == True

def test_sanitize():
    """Test recursive sanitization."""
    tainted_struct = mark_tainted({
        "a": [1, "unsafe"],
        "b": {"c": "dangerous"}
    })
    assert is_tainted(tainted_struct) == True

    clean_struct = sanitize(tainted_struct)
    assert is_tainted(clean_struct) == False
    assert clean_struct == {"a": [1, "unsafe"], "b": {"c": "dangerous"}}
    assert isinstance(clean_struct["a"][1], str)
    assert not isinstance(clean_struct["a"][1], str.__subclasses__()[0] if str.__subclasses__() else object) # Should be regular str

def test_taint_propagation_in_sandbox():
    """Test that taint propagates through function execution in sandbox."""
    kernel = MCPKernel()

    def identity(x):
        return x

    def join_strings(a, b):
        return f"{a} and {b}"

    tainted_val = mark_tainted("tainted")
    safe_val = "safe"

    # Propagate through identity
    res1 = kernel.sandbox.execute(identity, tainted_val)
    assert is_tainted(res1)

    # Propagate through join
    res2 = kernel.sandbox.execute(join_strings, safe_val, tainted_val)
    assert is_tainted(res2)

    res3 = kernel.sandbox.execute(join_strings, safe_val, safe_val)
    assert not is_tainted(res3)
