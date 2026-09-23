"""MCPKernel Taint Tracking Sandbox V3.

Provides a runtime sandbox wrapper for MCP tool execution that tracks tainted
inputs (e.g. untrusted user strings) and blocks them from reaching sensitive tool
arguments (like file paths or command payloads).
"""
import inspect
from typing import Any, Callable, Dict, Optional, Set, Union

from magda_agent.safety.taint_tracking_v2 import (
    PolicyViolationError,
    SandboxExecutionEnvironmentV2,
    TaintTrackerV2,
)

DEFAULT_SENSITIVE_ARG_NAMES: Set[str] = {
    "file_path",
    "filepath",
    "path",
    "command",
    "cmd",
    "payload",
    "script",
    "url",
    "code",
    "exec_cmd",
    "target_file",
    "destination",
}


class TaintTrackerV3(TaintTrackerV2):
    """Enhanced TaintTracker for V3 sandbox environment."""

    def __init__(self) -> None:
        """Initialize TaintTrackerV3."""
        super().__init__()


class MCPKernelSandboxV3:
    """Runtime sandbox wrapper for MCP tool execution with taint tracking.

    Tracks tainted input variables and blocks them from reaching sensitive tool
    arguments such as file paths, shell commands, or executable payloads.
    """

    def __init__(
        self,
        tracker: Optional[TaintTrackerV3] = None,
        sensitive_arg_names: Optional[Set[str]] = None,
    ) -> None:
        """Initialize MCPKernelSandboxV3.

        Args:
            tracker: Optional TaintTrackerV3 instance.
            sensitive_arg_names: Optional set of argument names to treat as sensitive.
        """
        self.tracker = tracker or TaintTrackerV3()
        self.sandbox = SandboxExecutionEnvironmentV2(self.tracker)
        if sensitive_arg_names is not None:
            self.sensitive_arg_names = set(sensitive_arg_names)
        else:
            self.sensitive_arg_names = set(DEFAULT_SENSITIVE_ARG_NAMES)

    def is_argument_sensitive(
        self, arg_name: str, custom_sensitive_args: Optional[Set[str]] = None
    ) -> bool:
        """Check if an argument name is classified as sensitive.

        Args:
            arg_name: Name of the argument to check.
            custom_sensitive_args: Optional set of custom sensitive argument names.

        Returns:
            True if the argument is sensitive, False otherwise.
        """
        all_sensitive = {name.lower() for name in self.sensitive_arg_names}
        if custom_sensitive_args:
            all_sensitive.update(name.lower() for name in custom_sensitive_args)
        return arg_name.lower() in all_sensitive

    def check_sensitive_arguments(
        self,
        inputs: Dict[str, Any],
        is_sensitive: bool = False,
        sensitive_args: Optional[Set[str]] = None,
    ) -> None:
        """Inspect tool input arguments and raise PolicyViolationError if tainted.

        Args:
            inputs: Dictionary of input arguments.
            is_sensitive: If True, any tainted input to the tool will be blocked.
            sensitive_args: Optional set of specific sensitive argument names for this execution.

        Raises:
            PolicyViolationError: If tainted data reaches sensitive arguments or tool.
        """
        if is_sensitive and self.tracker.is_tainted(inputs):
            origins = self.tracker.get_origins(inputs)
            raise PolicyViolationError(
                f"Tainted input passed to sensitive tool call from origins: {origins}"
            )

        for arg_name, arg_val in inputs.items():
            if self.is_argument_sensitive(arg_name, custom_sensitive_args=sensitive_args):
                if self.tracker.is_tainted(arg_val):
                    origins = self.tracker.get_origins(arg_val)
                    raise PolicyViolationError(
                        f"Tainted data passed to sensitive argument '{arg_name}' from origins: {origins}"
                    )

    def execute(
        self,
        tool_func: Callable[..., Any],
        inputs: Dict[str, Any],
        is_sensitive: bool = False,
        sensitive_args: Optional[Set[str]] = None,
    ) -> Any:
        """Executes a synchronous tool function in the sandbox.

        Args:
            tool_func: The tool function to execute.
            inputs: Dict of input arguments.
            is_sensitive: Whether the tool as a whole is sensitive.
            sensitive_args: Optional set of argument names considered sensitive.

        Returns:
            The tool result, tainted if inputs contained tainted non-sensitive data.

        Raises:
            PolicyViolationError: If tainted data reaches sensitive arguments.
            RuntimeError: If execution fails due to unexpected errors.
        """
        self.check_sensitive_arguments(
            inputs, is_sensitive=is_sensitive, sensitive_args=sensitive_args
        )

        try:
            return self.sandbox.execute(tool_func, **inputs)
        except PolicyViolationError:
            raise
        except Exception as e:
            raise RuntimeError(f"Sandbox execution failed: {str(e)}")

    async def execute_async(
        self,
        tool_func: Callable[..., Any],
        inputs: Dict[str, Any],
        is_sensitive: bool = False,
        sensitive_args: Optional[Set[str]] = None,
    ) -> Any:
        """Executes an asynchronous tool function in the sandbox.

        Args:
            tool_func: The async tool function to execute.
            inputs: Dict of input arguments.
            is_sensitive: Whether the tool as a whole is sensitive.
            sensitive_args: Optional set of argument names considered sensitive.

        Returns:
            The awaited tool result, tainted if inputs contained tainted non-sensitive data.

        Raises:
            PolicyViolationError: If tainted data reaches sensitive arguments.
            RuntimeError: If execution fails due to unexpected errors.
        """
        self.check_sensitive_arguments(
            inputs, is_sensitive=is_sensitive, sensitive_args=sensitive_args
        )

        try:
            return await self.sandbox.execute_async(tool_func, **inputs)
        except PolicyViolationError:
            raise
        except Exception as e:
            raise RuntimeError(f"Sandbox execution failed: {str(e)}")
