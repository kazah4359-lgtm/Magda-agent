"""Tests for ACS Guard Tool Sandboxing."""
import pytest
from magda_agent.safety.acs_sandboxing import ACSToolSandbox
from magda_agent.safety.taint_tracking_v2 import PolicyViolationError

def mock_read_tool(file_path: str) -> str:
    """A tool that reads data."""
    return f"Data from {file_path}"

def mock_write_tool(data: str) -> str:
    """A tool that writes data."""
    return "Write successful"

def mock_failing_tool(data: str) -> str:
    """A tool that fails."""
    raise ValueError("File not found")

def test_acs_tool_sandbox():
    """Test sandboxing of restricted tools."""
    sandbox = ACSToolSandbox()
    sandbox.restrict_tool("write_tool")

    # Safe data to read
    read_res = sandbox.execute_tool("read_tool", mock_read_tool, file_path="safe.txt")
    assert "Data from safe.txt" in read_res
    assert not sandbox.tracker.is_tainted(read_res)

    # Safe data to write
    write_res = sandbox.execute_tool("write_tool", mock_write_tool, data=read_res)
    assert write_res == "Write successful"

    # Tainted data
    tainted_input = sandbox.tracker.taint("malicious payload", origin="external_api")

    # Read tool is not restricted, so it can take tainted data and return tainted output
    read_res_tainted = sandbox.execute_tool("read_tool", mock_read_tool, file_path=tainted_input)
    assert sandbox.tracker.is_tainted(read_res_tainted)

    # Write tool is restricted, so it should block tainted input
    with pytest.raises(PolicyViolationError) as exc_info:
        sandbox.execute_tool("write_tool", mock_write_tool, data=read_res_tainted)

    assert "Tainted input to restricted tool 'write_tool'" in str(exc_info.value)
    assert "external_api" in str(exc_info.value)

def test_acs_tool_sandbox_execution_error():
    """Test that tool exceptions are wrapped properly."""
    sandbox = ACSToolSandbox()
    with pytest.raises(RuntimeError) as exc_info:
        sandbox.execute_tool("failing_tool", mock_failing_tool, data="test")
    assert "Tool execution failed: File not found" in str(exc_info.value)
