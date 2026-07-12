import pytest
import os
import tempfile
import asyncio
from typing import Any
from magda_agent.integration.mcp_discovery_v4 import MCPDynamicDiscoveryV4, MCPTool

@pytest.mark.asyncio
async def test_mcp_discovery_v4_hot_reload() -> None:
    """
    Test the hot reloading mechanism of MCPDynamicDiscoveryV4.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        discovery = MCPDynamicDiscoveryV4(temp_dir)

        # Initial scan, should be empty
        await discovery.scan_and_register()
        assert len(discovery.get_registered_tools()) == 0

        # Create a mock tool file
        tool_file_path = os.path.join(temp_dir, "mock_tool.py")
        with open(tool_file_path, "w") as f:
            f.write("""
from typing import Any
from magda_agent.integration.mcp_discovery_v4 import MCPTool

def mock_callable() -> None:
    pass

def register_tool() -> MCPTool:
    return MCPTool(name="mock_tool", description="A mock tool", callable_func=mock_callable)
""")

        # Scan again, should find the new tool
        await discovery.scan_and_register()
        tools = discovery.get_registered_tools()
        assert len(tools) == 1
        assert tools[0].name == "mock_tool"

        # Wait a bit to ensure mtime changes
        await asyncio.sleep(0.1)

        # Modify the tool file
        with open(tool_file_path, "w") as f:
             f.write("""
from typing import Any
from magda_agent.integration.mcp_discovery_v4 import MCPTool

def mock_callable_v2() -> None:
    pass

def register_tool() -> MCPTool:
    return MCPTool(name="mock_tool_v2", description="A modified mock tool", callable_func=mock_callable_v2)
""")

        # Scan again, should update the tool and remove the old one
        await discovery.scan_and_register()
        tools = discovery.get_registered_tools()
        assert len(tools) == 1
        assert tools[0].name == "mock_tool_v2"

        # Test file deletion
        os.remove(tool_file_path)
        await discovery.scan_and_register()
        tools = discovery.get_registered_tools()
        assert len(tools) == 0

        # Test ignoring files with no tools
        no_tool_file_path = os.path.join(temp_dir, "no_tool.py")
        with open(no_tool_file_path, "w") as f:
             f.write("""
def some_func() -> None:
    pass
""")
        await discovery.scan_and_register()
        tools = discovery.get_registered_tools()
        assert len(tools) == 0

        # Ensure mtime is tracked for the no_tool file so we don't infinitely re-load it
        assert no_tool_file_path in discovery._last_mtime
