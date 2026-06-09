"""Tests for the MCPKernel Taint Tracking Sandbox."""
import pytest
from typing import Dict, Any
from magda_agent.safety.taint import MCPKernel, PolicyViolationError

def sample_sensitive_tool(data: Dict[str, Any]) -> str:
    """A sample sensitive tool that returns processed data."""
    return f"Processed {data}"

def sample_safe_tool(data: Dict[str, Any]) -> str:
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

    # Safe tool should accept untainted
    res = kernel.execute_tool(sample_safe_tool, {"data": safe_data}, is_sensitive=False)
    assert "Safe" in res

    # Safe tool should accept tainted (not sensitive)
    res = kernel.execute_tool(sample_safe_tool, {"data": tainted_data}, is_sensitive=False)
    assert "Safe" in res

    # Sensitive tool should reject tainted
    with pytest.raises(PolicyViolationError) as exc_info:
        kernel.execute_tool(sample_sensitive_tool, {"data": tainted_string}, is_sensitive=True)
    assert "Tainted input to sensitive tool call" in str(exc_info.value)

    # Sensitive tool should accept safe data
    res = kernel.execute_tool(sample_sensitive_tool, {"data": safe_data}, is_sensitive=True)
    assert "Processed" in res
