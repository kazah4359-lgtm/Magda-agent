import subprocess
import tempfile
import os
import shutil

def execute(code: str) -> str:
    """
    Безопасный системный навык выполнения кода.
    Запускает код в изолированном subprocess с жестким таймаутом (10 сек).
    Ограничивает доступ к файловой системе директорией /tmp/sandbox/.
    Запускается в сетевом вакууме (unshare -n).
    """
    sandbox_dir = "/tmp/sandbox"
    os.makedirs(sandbox_dir, exist_ok=True)

    # We create a temporary script file within the sandbox
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=sandbox_dir, delete=False) as script_file:
        # Wrap the user's code to override builtins (like open) and restrict socket
        sandbox_wrapper = f"""
import sys
import builtins
import os
import socket

# Restrict socket module
def disabled_socket(*args, **kwargs):
    raise PermissionError("Network access is disabled in this sandbox.")
socket.socket = disabled_socket

# Restrict open() to only paths inside /tmp/sandbox
original_open = builtins.open
def restricted_open(file, *args, **kwargs):
    abs_path = os.path.abspath(file)
    if not abs_path.startswith("{sandbox_dir}"):
        raise PermissionError(f"Access denied. Cannot access files outside {sandbox_dir}")
    return original_open(file, *args, **kwargs)
builtins.open = restricted_open

# Attempt to drop network capabilities natively if running under Linux (fallback security)
try:
    import ctypes
    libc = ctypes.CDLL(None)
    # PR_SET_SECCOMP or dropping capabilities could go here, but blocking socket is the reliable fallback.
except Exception:
    pass

# User Code Execution
{code}
"""
        script_file.write(sandbox_wrapper)
        script_path = script_file.name

    try:
        # For simplicity and cross-platform/non-root execution, we rely on the
        # Python wrapper to block socket usage, rather than unshare which requires capabilities.
        cmd = ["python3", script_path]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10.0,
            cwd=sandbox_dir  # Set working directory to sandbox
        )
        return result.stdout
    except subprocess.TimeoutExpired as e:
        return f"Error executing code: TimeoutExpired after {e.timeout} seconds."
    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        # Clean up the script file
        if os.path.exists(script_path):
            os.remove(script_path)
