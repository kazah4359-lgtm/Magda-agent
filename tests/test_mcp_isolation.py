import pytest
from magda_agent.safety.mcp_isolation import MCPIsolationEngine
from magda_agent.safety.taint_tracking_v2 import PolicyViolationError

def test_mcp_isolation_engine_safe_execution() -> None:
    """Test that a safe tool executes successfully."""
    engine = MCPIsolationEngine()

    def safe_tool(data: str) -> str:
        return f"Processed: {data}"

    result = engine.execute_untrusted_tool(safe_tool, {"data": "clean_data"})
    assert result == "Processed: clean_data"

def test_mcp_isolation_engine_blocks_tainted_output() -> None:
    """Test that untrusted tool outputs are correctly blocked in isolation mode."""
    engine = MCPIsolationEngine()

    # Simulate a tool that receives tainted input
    def malicious_tool(data: str) -> str:
        return data

    tainted_input = engine.policy_engine.tracker.taint("secret_data", "source1")

    with pytest.raises(PolicyViolationError) as exc_info:
        engine.execute_untrusted_tool(malicious_tool, {"data": tainted_input})

    assert "Tainted output from untrusted tool call" in str(exc_info.value)
