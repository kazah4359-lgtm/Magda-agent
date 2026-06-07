from magda_agent.skills.mcp_kernel_executor import execute

def test_execute_safe_code() -> None:
    """Test execution skill with safe code."""
    res = execute("x = 5")
    assert "'x': 5" in res

def test_execute_unsafe_code() -> None:
    """Test execution skill with unsafe code."""
    res = execute("import os")
    assert "SecurityError" in res
