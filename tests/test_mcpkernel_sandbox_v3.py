"""Tests for MCPKernel Taint Tracking Sandbox V3."""
import pytest
from magda_agent.safety.mcpkernel_sandbox_v3 import (
    MCPKernelSandboxV3,
    TaintTrackerV3,
)
from magda_agent.safety.taint_tracking_v2 import PolicyViolationError


def sample_read_file(path: str) -> str:
    """Sample tool function that reads a file path."""
    return f"content of {path}"


def sample_run_command(cmd: str, timeout: int = 10) -> str:
    """Sample tool function that executes a command."""
    return f"executed {cmd} with timeout {timeout}"


def sample_process_text(text: str, label: str = "default") -> str:
    """Sample non-sensitive text processing tool."""
    return f"processed: {text} ({label})"


async def async_run_command(command: str) -> str:
    """Async sample tool function that executes a command."""
    return f"async executed {command}"


async def async_process_text(text: str) -> str:
    """Async sample non-sensitive text tool."""
    return f"async processed {text}"


def test_taint_tracker_v3_operations():
    """Verify TaintTrackerV3 functionality for tainting and origin tracking."""
    tracker = TaintTrackerV3()
    untainted_str = "user input"
    tainted_str = tracker.taint(untainted_str, "web_form")

    assert tracker.is_tainted(tainted_str)
    assert not tracker.is_tainted(untainted_str)
    assert tracker.get_origins(tainted_str) == {"web_form"}

    sanitized = tracker.sanitize(tainted_str)
    assert not tracker.is_tainted(sanitized)
    assert sanitized == "user input"


def test_sandbox_blocks_tainted_sensitive_arguments():
    """Verify sandbox blocks tainted strings passed to sensitive argument names."""
    tracker = TaintTrackerV3()
    sandbox = MCPKernelSandboxV3(tracker=tracker)

    tainted_path = tracker.taint("/etc/passwd", "user_prompt")
    inputs = {"path": tainted_path}

    with pytest.raises(PolicyViolationError) as exc_info:
        sandbox.execute(sample_read_file, inputs)

    assert "sensitive argument 'path'" in str(exc_info.value)
    assert "user_prompt" in str(exc_info.value)


def test_sandbox_blocks_tainted_command_payload():
    """Verify sandbox blocks command execution when cmd/command is tainted."""
    tracker = TaintTrackerV3()
    sandbox = MCPKernelSandboxV3(tracker=tracker)

    tainted_cmd = tracker.taint("rm -rf /", "malicious_user")
    inputs = {"cmd": tainted_cmd, "timeout": 5}

    with pytest.raises(PolicyViolationError) as exc_info:
        sandbox.execute(sample_run_command, inputs)

    assert "sensitive argument 'cmd'" in str(exc_info.value)
    assert "malicious_user" in str(exc_info.value)


def test_sandbox_allows_untainted_sensitive_arguments():
    """Verify sandbox executes successfully when sensitive arguments are untainted."""
    sandbox = MCPKernelSandboxV3()
    inputs = {"path": "/tmp/safe_file.txt"}

    result = sandbox.execute(sample_read_file, inputs)
    assert result == "content of /tmp/safe_file.txt"


def test_sandbox_allows_tainted_non_sensitive_arguments_and_propagates_taint():
    """Verify non-sensitive arguments allow tainted data and propagate taint to output."""
    tracker = TaintTrackerV3()
    sandbox = MCPKernelSandboxV3(tracker=tracker)

    tainted_text = tracker.taint("hello world", "untrusted_chat")
    inputs = {"text": tainted_text, "label": "test"}

    result = sandbox.execute(sample_process_text, inputs)

    assert tracker.is_tainted(result)
    assert tracker.get_origins(result) == {"untrusted_chat"}
    assert tracker.sanitize(result) == "processed: hello world (test)"


def test_sandbox_is_sensitive_flag_blocks_any_tainted_input():
    """Verify setting is_sensitive=True blocks any tainted input regardless of arg name."""
    tracker = TaintTrackerV3()
    sandbox = MCPKernelSandboxV3(tracker=tracker)

    tainted_val = tracker.taint("some_data", "external_api")
    inputs = {"text": tainted_val}

    with pytest.raises(PolicyViolationError) as exc_info:
        sandbox.execute(sample_process_text, inputs, is_sensitive=True)

    assert "sensitive tool call" in str(exc_info.value)


def test_custom_sensitive_args_parameter():
    """Verify passing custom sensitive_args to sandbox execution."""
    tracker = TaintTrackerV3()
    sandbox = MCPKernelSandboxV3(tracker=tracker)

    tainted_label = tracker.taint("secret_label", "user_input")
    inputs = {"text": "normal_text", "label": tainted_label}

    # Should succeed normally because label is not in default sensitive args
    res = sandbox.execute(sample_process_text, inputs)
    assert tracker.is_tainted(res)

    # Should fail when 'label' is explicitly passed in sensitive_args
    with pytest.raises(PolicyViolationError):
        sandbox.execute(sample_process_text, inputs, sensitive_args={"label"})


@pytest.mark.asyncio
async def test_async_execution_blocks_tainted_sensitive_arg():
    """Verify async tool execution blocks tainted sensitive arguments."""
    tracker = TaintTrackerV3()
    sandbox = MCPKernelSandboxV3(tracker=tracker)

    tainted_cmd = tracker.taint("ls -la", "remote_user")
    inputs = {"command": tainted_cmd}

    with pytest.raises(PolicyViolationError):
        await sandbox.execute_async(async_run_command, inputs)


@pytest.mark.asyncio
async def test_async_execution_allows_and_propagates_taint():
    """Verify async tool execution allows non-sensitive tainted args and propagates taint."""
    tracker = TaintTrackerV3()
    sandbox = MCPKernelSandboxV3(tracker=tracker)

    tainted_text = tracker.taint("async data", "web_socket")
    inputs = {"text": tainted_text}

    result = await sandbox.execute_async(async_process_text, inputs)

    assert tracker.is_tainted(result)
    assert tracker.get_origins(result) == {"web_socket"}


def test_sandbox_wraps_tool_exception():
    """Verify tool execution exceptions are wrapped in RuntimeError."""
    sandbox = MCPKernelSandboxV3()

    def failing_tool(path: str) -> None:
        raise ValueError("File not found")

    inputs = {"path": "/nonexistent"}

    with pytest.raises(RuntimeError) as exc_info:
        sandbox.execute(failing_tool, inputs)

    assert "Sandbox execution failed: File not found" in str(exc_info.value)
