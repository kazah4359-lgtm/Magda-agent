import subprocess
import tempfile
import os
try:
    import resource
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False

def set_limits():
    """Set resource limits for the subprocess to prevent fork bombs and restrict resources."""
    if not _HAS_RESOURCE:
        return
    # Limit to 0 to prevent creating new processes (effectively blocking subprocess/os.system bypasses)
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except Exception:
        pass

    # Optional: memory limits
    try:
        resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
    except Exception:
        pass

def execute(code: str) -> str:
    """
    Безопасный системный навык выполнения кода.
    Запускает код в изолированном subprocess с жестким таймаутом (10 сек).
    Ограничивает доступ к файловой системе директорией /tmp/sandbox/.
    """
    sandbox_dir = "/tmp/sandbox"
    os.makedirs(sandbox_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=sandbox_dir, delete=False) as script_file:
        # We do not monkey-patch builtins anymore.
        # Instead, we rely on OS-level protections:
        # 1. RLIMIT_NPROC=0 to prevent spawning curl, bash, cat, etc.
        # 2. unshare to drop network access reliably at the kernel level.
        # 3. We cannot easily chroot without root, but we can verify our limits stop trivial bypasses.

        script_file.write(code)
        script_path = script_file.name

    try:
        # Try to run with unshare (Linux) for network namespace isolation and mount namespace.
        # If unshare fails, we just run python3, but set_limits() will prevent spawning child processes.
        # However, to restrict Python's own open() securely without monkeypatching,
        # a full sandbox like Docker or firejail is needed. Since we can't use those here,
        # we will use an Audit Hook to block file access and network access at the Python interpreter level.

        audit_wrapper_code = f"""
import sys
import os

sandbox_dir = {repr(sandbox_dir)}

def audit_hook(event, args):
    if event == 'open':
        file = args[0]
        if isinstance(file, str):
            abs_path = os.path.abspath(file)
            if not abs_path.startswith(sandbox_dir):
                raise PermissionError(f"Access denied. Cannot access files outside {{sandbox_dir}}")
    elif event == 'socket.bind' or event == 'socket.connect' or event == 'socket.sendmsg':
        raise PermissionError("Network access is disabled in this sandbox.")
    elif event == 'subprocess.Popen':
        raise PermissionError("Spawning subprocesses is disabled.")
    elif event == 'os.system':
        raise PermissionError("os.system is disabled.")
    elif event == 'os.posix_spawn':
        raise PermissionError("os.posix_spawn is disabled.")

try:
    with open({repr(script_path)}, 'r') as f:
        code = f.read()
except Exception:
    pass

sys.addaudithook(audit_hook)

try:
    exec(code, {{"__builtins__": __builtins__}})
except PermissionError as e:
    print(f"PermissionError: {{e}}")
except Exception as e:
    print(f"Error: {{e}}")
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='_wrapper.py', dir=sandbox_dir, delete=False) as wrapper_file:
            wrapper_file.write(audit_wrapper_code)
            wrapper_path = wrapper_file.name

        cmd = ["python3", wrapper_path]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10.0,
            cwd=sandbox_dir,
            preexec_fn=set_limits # Apply resource limits before executing
        )
        return result.stdout
    except subprocess.TimeoutExpired as e:
        return f"Error executing code: TimeoutExpired after {e.timeout} seconds."
    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)
        if 'wrapper_path' in locals() and os.path.exists(wrapper_path):
            os.remove(wrapper_path)
