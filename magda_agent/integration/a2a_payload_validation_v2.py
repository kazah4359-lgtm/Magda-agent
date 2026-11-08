"""A2A Secure Payload Validation V2.

Inspired by A2A Protocol standard trends: Enhance payload validation to ensure
secure agent-to-agent communication payloads.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class PayloadValidationError(Exception):
    """Base exception for A2A payload validation errors."""
    pass


class PayloadStructureError(PayloadValidationError):
    """Raised when payload structure or required fields are missing or invalid."""
    pass


class PayloadSizeExceededError(PayloadValidationError):
    """Raised when payload size or nesting depth exceeds allowed limits."""
    pass


class PayloadSecurityViolationError(PayloadValidationError):
    """Raised when dangerous patterns or security policy violations are detected."""
    pass


class PayloadReplayError(PayloadValidationError):
    """Raised when timestamp drift indicates a replay attack or expired payload."""
    pass


class PayloadSignatureError(PayloadValidationError):
    """Raised when cryptographic signature validation fails."""
    pass


# Common dangerous injection and execution patterns
DANGEROUS_PATTERNS = [
    # Shell commands / deletion
    r"\brm\s+-[rf]{1,2}\b",
    r"\bmkfs\b",
    r"\bchmod\s+777\b",
    r"\bcurl\s+.*\|\s*(?:bash|sh)\b",
    r"\bwget\s+.*\|\s*(?:bash|sh)\b",
    # Code execution primitives
    r"\b__import__\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bos\.system\s*\(",
    r"\bsubprocess\.(?:Popen|run|call|check_output)\s*\(",
    # Path traversal
    r"(?:\.\./|\.\.\\){2,}",
    # Prompt injection patterns
    r"ignore\s+(?:all\s+)?previous\s+instructions",
    r"disregard\s+(?:all\s+)?prior\s+context",
    r"system\s*:\s*you\s+are\s+now",
    # SQL injection primitives
    r"\bdrop\s+table\b",
    r"\bunion\s+select\b",
    r"--\s*$",
    r";\s*drop\s+database\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_PATTERNS]


@dataclass
class A2APayloadV2:
    """Standardized and validated A2A communication payload."""

    task_id: str
    action: str
    sender_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    recipient_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    protocol_version: str = "v2"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the payload to a dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serializes the payload to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2APayloadV2":
        """Builds an A2APayloadV2 instance from a dictionary."""
        return cls(
            task_id=str(data.get("task_id", "")),
            action=str(data.get("action", data.get("method", ""))),
            sender_id=str(data.get("sender_id", "")),
            payload=data.get("payload", data.get("params", {})),
            recipient_id=data.get("recipient_id"),
            timestamp=float(data.get("timestamp", time.time())),
            signature=data.get("signature"),
            metadata=data.get("metadata", {}),
            protocol_version=str(data.get("protocol_version", "v2")),
        )


class A2ASecurePayloadValidatorV2:
    """Enhanced secure payload validator for A2A communications."""

    def __init__(
        self,
        max_payload_bytes: int = 1_000_000,  # 1MB limit
        max_nesting_depth: int = 15,
        max_timestamp_drift_seconds: float = 300.0,  # 5 minutes
        allowed_actions: Optional[Set[str]] = None,
        require_signature: bool = False,
        hmac_secret: Optional[str] = None,
        strict_security_scan: bool = True,
    ) -> None:
        self.max_payload_bytes = max_payload_bytes
        self.max_nesting_depth = max_nesting_depth
        self.max_timestamp_drift_seconds = max_timestamp_drift_seconds
        self.allowed_actions = allowed_actions
        self.require_signature = require_signature
        self.hmac_secret = hmac_secret
        self.strict_security_scan = strict_security_scan

    def check_size(self, raw_input: Union[str, bytes, Dict[str, Any]]) -> int:
        """Checks if the raw payload size is within limits.

        Returns:
            The size in bytes.
        """
        if isinstance(raw_input, bytes):
            size = len(raw_input)
        elif isinstance(raw_input, str):
            size = len(raw_input.encode("utf-8"))
        elif isinstance(raw_input, dict):
            size = len(json.dumps(raw_input).encode("utf-8"))
        else:
            raise PayloadStructureError(f"Payload must be dict, str, or bytes, got {type(raw_input)}")

        if size > self.max_payload_bytes:
            raise PayloadSizeExceededError(
                f"Payload size {size} bytes exceeds limit of {self.max_payload_bytes} bytes."
            )
        return size

    def check_nesting_depth(self, data: Any, current_depth: int = 0) -> None:
        """Checks if recursive data nesting exceeds max_nesting_depth."""
        if current_depth > self.max_nesting_depth:
            raise PayloadSizeExceededError(
                f"Payload nesting depth exceeded maximum allowable depth of {self.max_nesting_depth}."
            )
        if isinstance(data, dict):
            for v in data.values():
                self.check_nesting_depth(v, current_depth + 1)
        elif isinstance(data, (list, tuple, set)):
            for item in data:
                self.check_nesting_depth(item, current_depth + 1)

    def scan_for_dangerous_patterns(self, text_or_obj: Any) -> None:
        """Recursively scans all string fields for dangerous patterns or injections."""
        if isinstance(text_or_obj, str):
            for pattern in COMPILED_PATTERNS:
                if pattern.search(text_or_obj):
                    logger.warning(f"Malicious pattern match: {pattern.pattern} in text: {text_or_obj[:100]}")
                    raise PayloadSecurityViolationError(
                        f"Dangerous pattern detected in payload: {pattern.pattern}"
                    )
        elif isinstance(text_or_obj, dict):
            for k, v in text_or_obj.items():
                self.scan_for_dangerous_patterns(k)
                self.scan_for_dangerous_patterns(v)
        elif isinstance(text_or_obj, (list, tuple, set)):
            for item in text_or_obj:
                self.scan_for_dangerous_patterns(item)

    def verify_timestamp(self, ts: float) -> None:
        """Verifies timestamp freshness against replay attacks."""
        now = time.time()
        drift = abs(now - ts)
        if drift > self.max_timestamp_drift_seconds:
            raise PayloadReplayError(
                f"Payload timestamp drift of {drift:.2f}s exceeds allowable threshold of "
                f"{self.max_timestamp_drift_seconds}s."
            )

    @staticmethod
    def compute_signature(payload_dict: Dict[str, Any], secret: str) -> str:
        """Computes HMAC-SHA256 signature for a payload dictionary."""
        # Clean signature from dict before signing
        clean_copy = dict(payload_dict)
        clean_copy.pop("signature", None)
        canonical = json.dumps(clean_copy, sort_keys=True, separators=(",", ":"))
        return hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_signature(self, payload_dict: Dict[str, Any]) -> None:
        """Verifies cryptographic signature if secret is configured or required."""
        sig = payload_dict.get("signature")
        if not sig:
            if self.require_signature:
                raise PayloadSignatureError("Missing required signature in A2A payload.")
            return

        if not self.hmac_secret:
            logger.warning("Signature provided but no hmac_secret configured on validator.")
            return

        expected = self.compute_signature(payload_dict, self.hmac_secret)
        if not hmac.compare_digest(sig, expected):
            raise PayloadSignatureError("Invalid cryptographic signature in A2A payload.")

    def sanitize_payload(self, data: Any) -> Any:
        """Sanitizes text and nested fields, removing dangerous characters or stripping invalid controls."""
        if isinstance(data, str):
            # Strip null bytes and non-printable control characters except newline and tab
            cleaned = "".join(ch for ch in data if ch in ("\n", "\t", "\r") or (ord(ch) >= 32 and ord(ch) != 127))
            return cleaned.strip()
        elif isinstance(data, dict):
            return {self.sanitize_payload(k): self.sanitize_payload(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_payload(item) for item in data]
        return data

    def validate(self, raw_input: Union[str, bytes, Dict[str, Any]]) -> A2APayloadV2:
        """Fully validates an incoming payload and returns a typed A2APayloadV2 instance.

        Raises:
            PayloadValidationError or subclass on any validation failure.
        """
        # 1. Size check
        self.check_size(raw_input)

        # 2. Deserialization
        if isinstance(raw_input, (str, bytes)):
            try:
                data = json.loads(raw_input)
            except Exception as e:
                raise PayloadStructureError(f"Failed to parse payload as JSON: {e}")
        elif isinstance(raw_input, dict):
            data = raw_input
        else:
            raise PayloadStructureError(f"Payload must be dict, str, or bytes, got {type(raw_input)}")

        if not isinstance(data, dict):
            raise PayloadStructureError("A2A payload root must be a dictionary.")

        # 3. Depth check
        self.check_nesting_depth(data)

        # 4. Required fields check
        task_id = data.get("task_id")
        action = data.get("action") or data.get("method")
        sender_id = data.get("sender_id")

        if not task_id or not isinstance(task_id, str) or not task_id.strip():
            raise PayloadStructureError("Missing or empty required string field 'task_id'.")
        if not action or not isinstance(action, str) or not action.strip():
            raise PayloadStructureError("Missing or empty required string field 'action' or 'method'.")
        if not sender_id or not isinstance(sender_id, str) or not sender_id.strip():
            raise PayloadStructureError("Missing or empty required string field 'sender_id'.")

        inner_payload = data.get("payload") if "payload" in data else data.get("params")
        if inner_payload is None:
            inner_payload = {}
        elif not isinstance(inner_payload, dict):
            raise PayloadStructureError("Field 'payload' (or 'params') must be a dictionary.")

        # 5. Allowed actions check
        if self.allowed_actions is not None and action not in self.allowed_actions:
            raise PayloadSecurityViolationError(
                f"Action '{action}' is not in allowed actions: {sorted(self.allowed_actions)}"
            )

        # 6. Timestamp replay check
        if "timestamp" in data:
            try:
                ts = float(data["timestamp"])
                self.verify_timestamp(ts)
            except (ValueError, TypeError):
                raise PayloadStructureError(f"Invalid timestamp value: {data['timestamp']}")
        else:
            ts = time.time()

        # 7. Signature verification
        self.verify_signature(data)

        # 8. Strict security scanning for injections
        if self.strict_security_scan:
            self.scan_for_dangerous_patterns(data)

        return A2APayloadV2.from_dict({**data, "timestamp": ts, "action": action, "payload": inner_payload})

    def validate_and_sanitize(self, raw_input: Union[str, bytes, Dict[str, Any]]) -> A2APayloadV2:
        """Validates payload and returns an instance with sanitized fields."""
        payload = self.validate(raw_input)
        sanitized_inner = self.sanitize_payload(payload.payload)
        payload.payload = sanitized_inner
        return payload

    def is_valid(self, raw_input: Union[str, bytes, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """Non-throwing validation check returning (is_valid, error_message)."""
        try:
            self.validate(raw_input)
            return True, None
        except Exception as e:
            return False, str(e)

    async def validate_async(self, raw_input: Union[str, bytes, Dict[str, Any]]) -> A2APayloadV2:
        """Asynchronous validation wrapper for non-blocking workflows."""
        return self.validate(raw_input)


# Class aliases for compatibility and ease of discovery
A2APayloadValidationV2 = A2ASecurePayloadValidatorV2
A2ASecurePayloadValidator = A2ASecurePayloadValidatorV2
