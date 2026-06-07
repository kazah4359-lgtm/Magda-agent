import pytest
from magda_agent.safety.taint import mark_tainted, is_tainted, sanitize
from magda_agent.safety.runtime_guard import RuntimeGuard, SecurityException

def test_tainted_string():
    """Test creating and identifying tainted strings."""
    tainted = mark_tainted("rm -rf /")
    assert is_tainted(tainted)
    assert tainted == "rm -rf /"

    clean = "echo hello"
    assert not is_tainted(clean)

def test_sanitize():
    """Test sanitizing tainted strings."""
    tainted = mark_tainted("ls -la")
    clean = sanitize(tainted)
    assert not is_tainted(clean)
    assert clean == "ls -la"

def dummy_tool(cmd: str) -> str:
    """A dummy tool for testing."""
    return f"Executed: {cmd}"

def test_runtime_guard_clean():
    """Test RuntimeGuard with clean arguments."""
    result = RuntimeGuard.execute_safely(dummy_tool, cmd="echo hello")
    assert result == "Executed: echo hello"

def test_runtime_guard_tainted():
    """Test RuntimeGuard with tainted arguments."""
    tainted_cmd = mark_tainted("rm -rf /")
    with pytest.raises(SecurityException, match="tainted and unsafe"):
        RuntimeGuard.execute_safely(dummy_tool, cmd=tainted_cmd)
