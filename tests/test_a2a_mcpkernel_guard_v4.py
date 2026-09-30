"""Tests for A2AMCPKernelGuardV4 execution hooks and safety checkpoints."""

import pytest
from magda_agent.safety.a2a_mcpkernel_guard_v4 import (
    A2AMCPKernelGuard,
    A2AMCPKernelGuardV4,
)
from magda_agent.safety.mcpkernel_sandbox_v3 import TaintTrackerV3
from magda_agent.safety.taint_tracking_v2 import PolicyViolationError


def sample_tool_read_file(filepath: str) -> str:
    """Sample tool function that reads a file path."""
    return f"content of {filepath}"


def sample_tool_process_text(text: str, tag: str = "info") -> str:
    """Sample non-sensitive text processing tool function."""
    return f"processed: {text} [{tag}]"


def sample_tool_sensitive_output(text: str) -> str:
    """Sample tool function that outputs sensitive key patterns."""
    return f"result: api_key=12345_secret"


async def async_sample_tool_run_command(command: str) -> str:
    """Async sample tool function that executes a command."""
    return f"async output of {command}"


async def async_sample_tool_process(text: str) -> str:
    """Async sample non-sensitive tool function."""
    return f"async processed {text}"


def test_guard_alias_and_initialization():
    """Verify class initialization and backward-compatible alias."""
    guard = A2AMCPKernelGuardV4()
    assert isinstance(guard, A2AMCPKernelGuardV4)
    assert A2AMCPKernelGuard is A2AMCPKernelGuardV4


def test_evaluate_payload_clean():
    """Verify evaluate_payload succeeds for untainted safe action data."""
    guard = A2AMCPKernelGuardV4()
    passed, msg = guard.evaluate_payload(
        action_name="read",
        tool_name="sample_tool_read_file",
        kwargs={"filepath": "/tmp/safe.txt"},
    )
    assert passed is True
    assert "passed" in msg


def test_evaluate_payload_checkpoint_failure_unauthorized_intent():
    """Verify evaluate_payload fails when ACS checkpoint 2 fails."""
    guard = A2AMCPKernelGuardV4()
    passed, msg = guard.evaluate_payload(
        action_name="unauthorized_action",
        tool_name="sample_tool_read_file",
        kwargs={"filepath": "/tmp/safe.txt"},
    )
    assert passed is False
    assert "Checkpoint 2 Failed" in msg


def test_evaluate_payload_checkpoint_failure_forbidden_tool():
    """Verify evaluate_payload fails when tool is forbidden."""
    guard = A2AMCPKernelGuardV4()
    passed, msg = guard.evaluate_payload(
        action_name="execute",
        tool_name="forbidden_tool",
        kwargs={"filepath": "/tmp/safe.txt"},
    )
    assert passed is False
    assert "Checkpoint 3 Failed" in msg


def test_evaluate_payload_tainted_sensitive_argument():
    """Verify evaluate_payload fails when sensitive parameter contains tainted input."""
    tracker = TaintTrackerV3()
    guard = A2AMCPKernelGuardV4(tracker=tracker)

    tainted_path = tracker.taint("/etc/shadow", "peer_agent_1")
    passed, msg = guard.evaluate_payload(
        action_name="execute",
        tool_name="sample_tool_read_file",
        kwargs={"filepath": tainted_path},
    )
    assert passed is False
    assert "filepath" in msg
    assert "peer_agent_1" in msg


def test_execute_hook_success():
    """Verify execute_hook executes clean tool function successfully."""
    guard = A2AMCPKernelGuardV4()
    res = guard.execute_hook(
        tool_func=sample_tool_read_file,
        action_name="execute",
        tool_name="sample_tool_read_file",
        kwargs={"filepath": "/tmp/test.txt"},
    )
    assert res == "content of /tmp/test.txt"


def test_execute_hook_blocks_tainted_peer_parameter():
    """Verify execute_hook raises PolicyViolationError when peer passes tainted parameter."""
    tracker = TaintTrackerV3()
    guard = A2AMCPKernelGuardV4(tracker=tracker)

    tainted_file = tracker.taint("/etc/passwd", "untrusted_peer_node")

    with pytest.raises(PolicyViolationError) as exc_info:
        guard.execute_hook(
            tool_func=sample_tool_read_file,
            action_name="execute",
            tool_name="sample_tool_read_file",
            kwargs={"filepath": tainted_file},
        )

    assert "A2A execution blocked" in str(exc_info.value) or "filepath" in str(
        exc_info.value
    )


def test_execute_hook_blocks_when_is_sensitive_is_true():
    """Verify execute_hook blocks any tainted input when is_sensitive=True."""
    tracker = TaintTrackerV3()
    guard = A2AMCPKernelGuardV4(tracker=tracker)

    tainted_text = tracker.taint("hello world", "peer_payload")

    with pytest.raises(PolicyViolationError) as exc_info:
        guard.execute_hook(
            tool_func=sample_tool_process_text,
            action_name="execute",
            tool_name="sample_tool_process_text",
            kwargs={"text": tainted_text},
            is_sensitive=True,
        )

    assert "blocked" in str(exc_info.value) or "sensitive" in str(exc_info.value)


def test_execute_hook_propagates_taint_for_non_sensitive_args():
    """Verify execute_hook allows non-sensitive tainted args and propagates taint to output."""
    tracker = TaintTrackerV3()
    guard = A2AMCPKernelGuardV4(tracker=tracker)

    tainted_text = tracker.taint("peer message", "external_agent")
    result = guard.execute_hook(
        tool_func=sample_tool_process_text,
        action_name="execute",
        tool_name="sample_tool_process_text",
        kwargs={"text": tainted_text, "tag": "v4"},
    )

    assert tracker.is_tainted(result)
    assert tracker.get_origins(result) == {"external_agent"}


def test_execute_hook_fails_post_execution_sanitization():
    """Verify post-execution checkpoint blocks output containing sensitive patterns."""
    guard = A2AMCPKernelGuardV4()

    with pytest.raises(PolicyViolationError) as exc_info:
        guard.execute_hook(
            tool_func=sample_tool_sensitive_output,
            action_name="execute",
            tool_name="sample_tool_sensitive_output",
            kwargs={"text": "clean_input"},
        )

    assert "A2A output evaluation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_execute_hook_async_success():
    """Verify execute_hook_async executes clean async tool function successfully."""
    guard = A2AMCPKernelGuardV4()
    res = await guard.execute_hook_async(
        tool_func=async_sample_tool_run_command,
        action_name="execute",
        tool_name="async_sample_tool_run_command",
        kwargs={"command": "echo 'hello'"},
    )
    assert res == "async output of echo 'hello'"


@pytest.mark.asyncio
async def test_execute_hook_async_blocks_tainted_peer_parameter():
    """Verify execute_hook_async blocks async execution with tainted peer parameters."""
    tracker = TaintTrackerV3()
    guard = A2AMCPKernelGuardV4(tracker=tracker)

    tainted_cmd = tracker.taint("rm -rf /", "malicious_peer")

    with pytest.raises(PolicyViolationError):
        await guard.execute_hook_async(
            tool_func=async_sample_tool_run_command,
            action_name="execute",
            tool_name="async_sample_tool_run_command",
            kwargs={"command": tainted_cmd},
        )
