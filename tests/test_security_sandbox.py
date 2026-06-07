import pytest
import os
import time
from magda_agent.skills.system_execute_code import execute

def test_successful_execution():
    """1. Test successful execution of clean, safe Python arithmetic/logic."""
    code = "print(10 * 5)\nprint('hello world')"
    result = execute(code)
    assert "50" in result
    assert "hello world" in result

def test_infinite_loop_timeout():
    """2. Test that an infinite loop is successfully killed exactly at 10 seconds."""
    code = "while True: pass"
    start_time = time.time()
    result = execute(code)
    elapsed = time.time() - start_time

    # Check that it took approximately 10 seconds
    assert 9.5 < elapsed < 11.5
    assert "TimeoutExpired after 10.0 seconds" in result

def test_bypass_attempt():
    """Verify that OS-level bypasses via os.system or subprocess are blocked by audit hook."""
    code = "import os\nos.system('echo bypassed')"
    result = execute(code)
    assert "PermissionError" in result
    assert "os.system is disabled" in result

def test_file_isolation_blocked():
    """3. Test that attempting to read sensitive system files is blocked."""
    code = "open('/etc/passwd', 'r').read()"
    result = execute(code)
    assert "PermissionError" in result
    assert "Access denied" in result

def test_file_isolation_allowed():
    """Test that writing/reading inside the sandbox is allowed."""
    code = "with open('/tmp/sandbox/test_allowed.txt', 'w') as f: f.write('secret')\nwith open('/tmp/sandbox/test_allowed.txt', 'r') as f: print(f.read())"
    result = execute(code)
    assert "secret" in result

def test_network_isolation():
    """4. Test that attempting to import socket and connect fails."""
    code = "import socket\ns = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\ns.connect(('8.8.8.8', 53))"
    result = execute(code)
    assert "PermissionError" in result
    # Python internal networking triggers getaddrinfo and open(/etc/resolv.conf) which fails on file sandbox first.
    # As long as it throws PermissionError, it is properly isolated.

def test_ctypes_import_bypass_is_blocked():
    """ctypes must not be importable because it can bypass Python audit hooks."""
    code = "import ctypes\nctypes.CDLL(None).system(b'echo bypassed')"
    result = execute(code)
    assert "PermissionError" in result
    assert "imports are disabled" in result
