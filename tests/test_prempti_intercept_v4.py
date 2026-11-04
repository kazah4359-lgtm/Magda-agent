import sys
import asyncio
import pytest
from typing import Any, Dict

from magda_agent.safety.prempti_intercept_v4 import (
    PremptiActionInterceptV4,
    PremptiAuditLoggerV4,
    PremptiPIIRedactorV4,
)


def test_pii_redactor_text_patterns() -> None:
    """Test text redaction across standard PII categories."""
    redactor = PremptiPIIRedactorV4

    # Email
    assert redactor.redact_text("Contact user@example.com for info") == "Contact [REDACTED_EMAIL] for info"
    assert redactor.redact_text("Emails: test.123+tag@domain.co.uk and simple@sub.org") == (
        "Emails: [REDACTED_EMAIL] and [REDACTED_EMAIL]"
    )

    # SSN
    assert redactor.redact_text("Social security is 123-45-6789.") == "Social security is [REDACTED_SSN]."

    # Phone numbers
    assert redactor.redact_text("Call +1-800-555-0199 now") == "Call [REDACTED_PHONE] now"
    assert redactor.redact_text("Reach us at (555) 123-4567") == "Reach us at [REDACTED_PHONE]"
    assert redactor.redact_text("Dial 555-123-4567 directly") == "Dial [REDACTED_PHONE] directly"

    # Credit Card
    assert redactor.redact_text("Card number 4111-2222-3333-4444 used") == "Card number [REDACTED_CREDIT_CARD] used"
    assert redactor.redact_text("Card 5500 0000 0000 0004 charged") == "Card [REDACTED_CREDIT_CARD] charged"

    # IPv4
    assert redactor.redact_text("Server connected from 192.168.1.100") == "Server connected from [REDACTED_IP]"

    # Bearer / API token
    assert redactor.redact_text("Authorization: Bearer abcdef1234567890abcdef1234567890") == (
        "Authorization: Bearer [REDACTED_SECRET]"
    )
    assert redactor.redact_text("api_key = 'secret12345678'") == "api_key = [REDACTED_SECRET]"
    assert redactor.redact_text("token: secret_token_xyz_12345") == "token: [REDACTED_SECRET]"


def test_pii_redactor_nested_data_sanitization() -> None:
    """Test recursive data structure sanitization including sensitive keys."""
    redactor = PremptiPIIRedactorV4

    payload = {
        "user_email": "john.doe@example.org",
        "nested": {
            "phone": "+1-555-555-5555",
            "password": "ClearTextPassword123!",
            "api_key": "sk-12345678901234567890",
            "safe_metadata": "Everything is running smoothly",
        },
        "records": [
            "User IP: 10.0.0.1",
            {"ssn": "000-12-3456", "note": "verified identity"}
        ],
        "safe_int": 42,
        "author": "Alice Author",  # 'auth' substring in 'author' shouldn't be redacted as a secret
        "environment": "production"  # 'env' substring in 'environment' shouldn't be redacted
    }

    sanitized = redactor.sanitize_data(payload)

    assert sanitized["user_email"] == "[REDACTED_EMAIL]"
    assert sanitized["nested"]["phone"] == "[REDACTED_PHONE]"
    assert sanitized["nested"]["password"] == "[REDACTED_SECRET]"
    assert sanitized["nested"]["api_key"] == "[REDACTED_SECRET]"
    assert sanitized["nested"]["safe_metadata"] == "Everything is running smoothly"
    assert sanitized["records"][0] == "User IP: [REDACTED_IP]"
    assert sanitized["records"][1]["ssn"] == "[REDACTED_SECRET]"
    assert sanitized["records"][1]["note"] == "verified identity"
    assert sanitized["safe_int"] == 42
    assert sanitized["author"] == "Alice Author"
    assert sanitized["environment"] == "production"


def test_circular_reference_sanitization() -> None:
    """Ensure circular references do not cause recursion errors during sanitization."""
    redactor = PremptiPIIRedactorV4
    data: Dict[str, Any] = {"name": "test"}
    data["self"] = data

    sanitized = redactor.sanitize_data(data)
    assert sanitized["name"] == "test"
    assert sanitized["self"] == "<circular reference>"


def test_sync_execution_stdout_redaction() -> None:
    """Test that synchronous tool execution intercepts stdout and masks PII."""
    interceptor = PremptiActionInterceptV4(pass_through_stream=False)

    @interceptor.intercept(tool_name="process_user_data")
    def sync_skill(name: str, email: str, secret_token: str) -> str:
        print(f"Debug: Processing {name} with contact {email} and token: {secret_token}")
        sys.stderr.write("Notice: User SSN is 111-22-3333\n")
        return f"Completed for {name} ({email})"

    result = sync_skill(
        name="Bob",
        email="bob@corp.com",
        secret_token="tok_1234567890"
    )

    # Return value must have PII masked in the audit log, though raw function returned
    assert result == "Completed for Bob (bob@corp.com)"

    logs = interceptor.get_logs(tool_name="process_user_data")
    assert len(logs) == 1
    log = logs[0]

    assert log["tool_name"] == "process_user_data"
    assert log["status"] == "success"
    assert log["duration"] >= 0.0

    # Verify stdout was captured and redacted
    assert "[REDACTED_EMAIL]" in log["stdout"]
    assert "bob@corp.com" not in log["stdout"]
    assert "[REDACTED_SECRET]" in log["stdout"]
    assert "tok_1234567890" not in log["stdout"]

    # Verify stderr was captured and redacted
    assert "[REDACTED_SSN]" in log["stderr"]
    assert "111-22-3333" not in log["stderr"]

    # Verify kwargs were sanitized
    assert log["kwargs"]["name"] == "Bob"
    assert log["kwargs"]["email"] == "[REDACTED_EMAIL]"
    assert log["kwargs"]["secret_token"] == "[REDACTED_SECRET]"

    # Verify result in log is sanitized
    assert log["result"] == "Completed for Bob ([REDACTED_EMAIL])"


@pytest.mark.asyncio
async def test_async_execution_stdout_redaction() -> None:
    """Test that asynchronous tool execution intercepts stdout and masks PII."""
    interceptor = PremptiActionInterceptV4(pass_through_stream=False)

    @interceptor.intercept(tool_name="fetch_customer_details")
    async def async_skill(customer_id: str, card: str) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        print(f"Customer {customer_id} card charged: {card} from IP 192.168.1.50")
        return {
            "status": "billed",
            "card_number": card,
            "ip": "192.168.1.50"
        }

    res = await async_skill("cust_999", card="4111-2222-3333-4444")
    assert res["status"] == "billed"

    logs = interceptor.get_logs("fetch_customer_details")
    assert len(logs) == 1
    log = logs[0]

    # Verify stdout PII redaction
    assert "[REDACTED_CREDIT_CARD]" in log["stdout"]
    assert "4111-2222-3333-4444" not in log["stdout"]
    assert "[REDACTED_IP]" in log["stdout"]
    assert "192.168.1.50" not in log["stdout"]

    # Verify logged result sanitization
    assert log["result"]["status"] == "billed"
    assert log["result"]["card_number"] == "[REDACTED_SECRET]"
    assert log["result"]["ip"] == "[REDACTED_IP]"


def test_execute_and_execute_async_helpers() -> None:
    """Test the direct execute and execute_async methods on PremptiActionInterceptV4."""
    interceptor = PremptiActionInterceptV4(pass_through_stream=False)

    def print_info(phone: str) -> str:
        print(f"Target phone: {phone}")
        return f"Sent to {phone}"

    res = interceptor.execute(print_info, phone="+1-202-555-0143", tool_name="sms_sender")
    assert res == "Sent to +1-202-555-0143"

    logs = interceptor.get_logs("sms_sender")
    assert len(logs) == 1
    assert "[REDACTED_PHONE]" in logs[0]["stdout"]
    assert "+1-202-555-0143" not in logs[0]["stdout"]


@pytest.mark.asyncio
async def test_execute_async_helper() -> None:
    """Test execute_async helper."""
    interceptor = PremptiActionInterceptV4(pass_through_stream=False)

    async def async_worker(email: str) -> str:
        print(f"Async email: {email}")
        return "ok"

    res = await interceptor.execute_async(async_worker, email="support@agent.ai", tool_name="email_worker")
    assert res == "ok"

    logs = interceptor.get_logs("email_worker")
    assert len(logs) == 1
    assert "[REDACTED_EMAIL]" in logs[0]["stdout"]
    assert "support@agent.ai" not in logs[0]["stdout"]


def test_exception_handling_in_interception() -> None:
    """Test that exceptions raised inside intercepted skills are logged and PII in error is redacted."""
    interceptor = PremptiActionInterceptV4(pass_through_stream=False)

    @interceptor.intercept(tool_name="failing_skill")
    def fail_with_pii(ssn: str) -> None:
        print(f"About to fail on SSN: {ssn}")
        raise ValueError(f"Failed processing record for SSN 999-88-7777")

    with pytest.raises(ValueError, match="Failed processing record for SSN 999-88-7777"):
        fail_with_pii("999-88-7777")

    logs = interceptor.get_logs("failing_skill")
    assert len(logs) == 1
    log = logs[0]
    assert log["status"] == "failed"
    assert "[REDACTED_SSN]" in log["stdout"]
    assert "999-88-7777" not in log["stdout"]
    assert "[REDACTED_SSN]" in log["error"]
    assert "999-88-7777" not in log["error"]


def test_alias_compatibility_and_clear() -> None:
    """Test PremptiAuditLoggerV4 alias and clear function."""
    logger = PremptiAuditLoggerV4(pass_through_stream=False)

    @logger.intercept(tool_name="dummy_tool")
    def dummy() -> str:
        print("Hello world")
        return "done"

    dummy()
    assert len(logger.get_logs()) == 1
    logger.clear()
    assert len(logger.get_logs()) == 0
