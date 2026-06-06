import pytest
from magda_agent.security.mcp_kernel import MCPKernel, SecurityError

def test_mcp_kernel_safe_execution() -> None:
    """Test safe execution of simple code block."""
    kernel = MCPKernel()
    code = "x = 5\ny = x + 10"
    locals_dict = {}
    kernel.execute(code, locals_dict=locals_dict)
    assert locals_dict.get('x') == 5
    assert locals_dict.get('y') == 15

def test_mcp_kernel_blocks_unsafe_call() -> None:
    """Test blocking of unsafe functions."""
    kernel = MCPKernel()
    code = "file = open('/etc/passwd', 'r')"
    with pytest.raises(SecurityError, match="Code contains unsafe operations"):
        kernel.execute(code)

def test_mcp_kernel_blocks_imports() -> None:
    """Test blocking of import statements."""
    kernel = MCPKernel()
    code = "import os\nos.system('echo hi')"
    with pytest.raises(SecurityError, match="Code contains unsafe operations"):
        kernel.execute(code)

def test_mcp_kernel_blocks_unallowed_builtins() -> None:
    """Test blocking of functions not in allowed set."""
    kernel = MCPKernel()
    # 'id' is a builtin but not in our default allowed set
    code = "x = id(5)"
    with pytest.raises(SecurityError, match="Code contains unsafe operations"):
        kernel.execute(code)

def test_mcp_kernel_blocks_attribute_bypass() -> None:
    """Test blocking of methods called on strings that might be unsafe."""
    kernel = MCPKernel()
    code = "''.__class__.__mro__[1].__subclasses__()"
    with pytest.raises(SecurityError, match="Code contains unsafe operations"):
        kernel.execute(code)
