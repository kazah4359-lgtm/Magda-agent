with open('tests/test_mcpkernel_taint.py', 'r') as f:
    content = f.read()

# Remove the import from the middle
content = content.replace("from magda_agent.security.mcp_kernel import MCPKernel, SecurityError\n\n", "")

# Add it to the top after import pytest
content = content.replace("import pytest\n", "import pytest\nfrom magda_agent.security.mcp_kernel import MCPKernel, SecurityError\n")

# Add the return type hint to the test function
content = content.replace("def test_mcp_kernel_execute_tainted():", "def test_mcp_kernel_execute_tainted() -> None:")

with open('tests/test_mcpkernel_taint.py', 'w') as f:
    f.write(content)
