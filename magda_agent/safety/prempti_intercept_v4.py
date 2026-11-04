import io
import re
import sys
import time
import copy
import inspect
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


class PremptiPIIRedactorV4:
    """
    Utility for detecting and masking Personally Identifiable Information (PII)
    and sensitive credentials in strings and arbitrary nested data structures.
    """

    # Common PII Regex Patterns
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )
    # Common Phone patterns: +1-234-567-8901, (123) 456-7890, 123-456-7890, etc.
    PHONE_PATTERN = re.compile(
        r"(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4})\b"
    )
    # US SSN pattern: 000-00-0000
    SSN_PATTERN = re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    )
    # Credit Card pattern (13-19 digits, optional dashes/spaces)
    CREDIT_CARD_PATTERN = re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{1,7}\b"
    )
    # IPv4 Pattern
    IPV4_PATTERN = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )
    # Bearer / Token / API Key pattern
    BEARER_PATTERN = re.compile(
        r"(?i)\b(bearer\s+[A-Za-z0-9_\-\.]{16,}|(?:api[_-]?key|token|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{8,}['\"]?)"
    )

    # Sensitive key patterns in dicts
    SENSITIVE_KEY_STEMS = {
        "password", "secret", "key", "token", "auth", "credential",
        "env", "api_key", "access_key", "private", "private_key",
        "ssn", "credit_card", "card_number", "cvv"
    }

    @classmethod
    def redact_text(cls, text: str) -> str:
        """
        Scans a text string and replaces detected PII and credentials with masked tokens.
        """
        if not isinstance(text, str) or not text:
            return text

        masked = text

        # Redact Bearer / API Keys first
        def mask_bearer(match: re.Match) -> str:
            val = match.group(0)
            if ":" in val or "=" in val:
                prefix = re.split(r"[:=]", val)[0]
                delimiter = ":" if ":" in val else "="
                return f"{prefix}{delimiter} [REDACTED_SECRET]"
            return "Bearer [REDACTED_SECRET]"

        masked = cls.BEARER_PATTERN.sub(mask_bearer, masked)

        # Redact SSN
        masked = cls.SSN_PATTERN.sub("[REDACTED_SSN]", masked)

        # Redact Credit Cards (check length of digits >= 13)
        def mask_cc(match: re.Match) -> str:
            digits = re.sub(r"\D", "", match.group(0))
            if 13 <= len(digits) <= 19:
                return "[REDACTED_CREDIT_CARD]"
            return match.group(0)

        masked = cls.CREDIT_CARD_PATTERN.sub(mask_cc, masked)

        # Redact Emails
        masked = cls.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", masked)

        # Redact IPv4 (excluding localhost if preferred, but general masking is safer)
        masked = cls.IPV4_PATTERN.sub("[REDACTED_IP]", masked)

        # Redact Phone Numbers (validate digits count >= 10 to avoid small numbers)
        def mask_phone(match: re.Match) -> str:
            digits = re.sub(r"\D", "", match.group(0))
            if len(digits) >= 10:
                return "[REDACTED_PHONE]"
            return match.group(0)

        masked = cls.PHONE_PATTERN.sub(mask_phone, masked)

        return masked

    @classmethod
    def sanitize_data(cls, data: Any, memo: Optional[Dict[int, Any]] = None) -> Any:
        """
        Recursively sanitizes sensitive data and PII from dictionaries, lists, and strings.
        Prevents circular reference loops with a memo dictionary.
        """
        if memo is None:
            memo = {}

        if id(data) in memo:
            return "<circular reference>"

        if inspect.isawaitable(data):
            return "<awaitable>"

        if isinstance(data, str):
            return cls.redact_text(data)

        if isinstance(data, dict):
            memo[id(data)] = True
            sanitized: Dict[str, Any] = {}
            for k, v in data.items():
                k_str = str(k).lower()
                is_sensitive = False
                for s in cls.SENSITIVE_KEY_STEMS:
                    if re.search(rf"(^|[^a-zA-Z]){re.escape(s)}s?([^a-zA-Z]|$)", k_str):
                        is_sensitive = True
                        break

                if is_sensitive:
                    sanitized[k] = "[REDACTED_SECRET]"
                else:
                    sanitized[k] = cls.sanitize_data(v, memo)
            return sanitized

        elif isinstance(data, (list, tuple, set)):
            memo[id(data)] = True
            sanitized_items = [cls.sanitize_data(item, memo) for item in data]
            if isinstance(data, tuple):
                return tuple(sanitized_items)
            elif isinstance(data, set):
                return set(sanitized_items)
            return sanitized_items

        try:
            return copy.deepcopy(data)
        except Exception:
            return repr(data)


class _RedactingStream(io.TextIOBase):
    """
    A stream wrapper that intercepts stdout/stderr writes, redacts PII in real-time,
    and forwards the sanitized text to the underlying target stream while also
    capturing it into an internal buffer.
    """

    def __init__(self, target_stream: Any, redactor: Callable[[str], str]) -> None:
        self._target_stream = target_stream
        self._redactor = redactor
        self._captured = io.StringIO()

    def write(self, s: str) -> int:
        if not s:
            return 0
        masked = self._redactor(s)
        self._captured.write(masked)
        if self._target_stream:
            self._target_stream.write(masked)
        return len(s)

    def flush(self) -> None:
        if self._target_stream and hasattr(self._target_stream, "flush"):
            self._target_stream.flush()

    def get_captured(self) -> str:
        return self._captured.getvalue()


class PremptiActionInterceptV4:
    """
    Action interception layer inspired by Prempti (Falco).
    Intercepts standard output during skill/tool execution to mask PII in real-time,
    scrubs function arguments and return values, and logs structured audit entries.
    """

    def __init__(self, pass_through_stream: bool = True) -> None:
        """
        Args:
            pass_through_stream: If True, writes redacted output to original sys.stdout/sys.stderr.
                                 If False, captures stdout/stderr silently without emitting.
        """
        self.pass_through_stream = pass_through_stream
        self.audit_log: List[Dict[str, Any]] = []

    def redact(self, text_or_data: Any) -> Any:
        """Convenience method to redact PII in text or nested data structures."""
        if isinstance(text_or_data, str):
            return PremptiPIIRedactorV4.redact_text(text_or_data)
        return PremptiPIIRedactorV4.sanitize_data(text_or_data)

    @contextmanager
    def capture_and_redact(self) -> Iterator[Dict[str, Any]]:
        """
        Context manager that intercepts sys.stdout and sys.stderr,
        redacting PII from any stream output produced during the block.
        Yields a dict that will contain {'stdout': captured_stdout, 'stderr': captured_stderr}.
        """
        captured: Dict[str, Any] = {"stdout": "", "stderr": ""}
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        out_target = old_stdout if self.pass_through_stream else None
        err_target = old_stderr if self.pass_through_stream else None

        redacting_stdout = _RedactingStream(out_target, PremptiPIIRedactorV4.redact_text)
        redacting_stderr = _RedactingStream(err_target, PremptiPIIRedactorV4.redact_text)

        sys.stdout = redacting_stdout
        sys.stderr = redacting_stderr
        try:
            yield captured
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            captured["stdout"] = redacting_stdout.get_captured()
            captured["stderr"] = redacting_stderr.get_captured()

    def _extract_args(
        self, func: Callable[..., Any], args: tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extracts and binds arguments passed to a function using its signature."""
        try:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return dict(bound.arguments)
        except Exception:
            extracted: Dict[str, Any] = {}
            for i, v in enumerate(args):
                extracted[f"arg_{i}"] = v
            extracted.update(kwargs)
            return extracted

    def log_action(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
        stdout: str,
        stderr: str,
        result: Any,
        duration: float,
        status: str = "success",
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Records a sanitized audit log entry."""
        entry = {
            "timestamp": time.time(),
            "tool_name": tool_name,
            "status": status,
            "kwargs": PremptiPIIRedactorV4.sanitize_data(kwargs),
            "stdout": stdout,
            "stderr": stderr,
            "result": PremptiPIIRedactorV4.sanitize_data(result),
            "duration": duration,
            "error": PremptiPIIRedactorV4.redact_text(error) if error else None,
        }
        self.audit_log.append(entry)
        return entry

    def intercept(
        self,
        tool_name: Optional[str] = None,
        why: str = "intercepted action"
    ) -> Callable[[F], F]:
        """
        Decorator to intercept a skill or action execution, capturing and masking
        stdout/stderr output, arguments, and return values.
        """
        def decorator(func: F) -> F:
            name_to_use = tool_name or getattr(func, "__name__", "anonymous_tool")

            if inspect.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    all_args = self._extract_args(func, args, kwargs)
                    start_time = time.time()
                    stdout_captured = ""
                    stderr_captured = ""
                    try:
                        with self.capture_and_redact() as stream_info:
                            result = await func(*args, **kwargs)
                        duration = time.time() - start_time
                        stdout_captured = stream_info.get("stdout", "")
                        stderr_captured = stream_info.get("stderr", "")

                        self.log_action(
                            tool_name=name_to_use,
                            kwargs=all_args,
                            stdout=stdout_captured,
                            stderr=stderr_captured,
                            result=result,
                            duration=duration,
                            status="success"
                        )
                        return result
                    except BaseException as e:
                        duration = time.time() - start_time
                        stdout_captured = stream_info.get("stdout", "")
                        stderr_captured = stream_info.get("stderr", "")
                        self.log_action(
                            tool_name=name_to_use,
                            kwargs=all_args,
                            stdout=stdout_captured,
                            stderr=stderr_captured,
                            result=None,
                            duration=duration,
                            status="failed",
                            error=str(e)
                        )
                        raise

                return cast(F, async_wrapper)
            else:
                @wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    all_args = self._extract_args(func, args, kwargs)
                    start_time = time.time()
                    stdout_captured = ""
                    stderr_captured = ""
                    try:
                        with self.capture_and_redact() as stream_info:
                            result = func(*args, **kwargs)
                        duration = time.time() - start_time
                        stdout_captured = stream_info.get("stdout", "")
                        stderr_captured = stream_info.get("stderr", "")

                        self.log_action(
                            tool_name=name_to_use,
                            kwargs=all_args,
                            stdout=stdout_captured,
                            stderr=stderr_captured,
                            result=result,
                            duration=duration,
                            status="success"
                        )
                        return result
                    except BaseException as e:
                        duration = time.time() - start_time
                        stdout_captured = stream_info.get("stdout", "")
                        stderr_captured = stream_info.get("stderr", "")
                        self.log_action(
                            tool_name=name_to_use,
                            kwargs=all_args,
                            stdout=stdout_captured,
                            stderr=stderr_captured,
                            result=None,
                            duration=duration,
                            status="failed",
                            error=str(e)
                        )
                        raise

                return cast(F, sync_wrapper)

        return decorator

    def execute(self, func: Callable[..., Any], *args: Any, tool_name: Optional[str] = None, **kwargs: Any) -> Any:
        """Executes a synchronous function under interception."""
        decorated = self.intercept(tool_name=tool_name)(func)
        return decorated(*args, **kwargs)

    async def execute_async(self, func: Callable[..., Any], *args: Any, tool_name: Optional[str] = None, **kwargs: Any) -> Any:
        """Executes an asynchronous function under interception."""
        decorated = self.intercept(tool_name=tool_name)(func)
        return await decorated(*args, **kwargs)

    def get_logs(self, tool_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns the audit log entries, optionally filtered by tool name."""
        if tool_name:
            return [e for e in self.audit_log if e["tool_name"] == tool_name]
        return list(self.audit_log)

    def clear(self) -> None:
        """Clears the internal audit log."""
        self.audit_log.clear()


# Alias for compatibility with common naming conventions
PremptiAuditLoggerV4 = PremptiActionInterceptV4
