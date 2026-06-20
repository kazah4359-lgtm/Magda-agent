import pytest
from magda_agent.skills.registry import SkillRegistry
from magda_agent.safety.acs_guard_v2 import SecurityViolationError

def test_skill_registry_audit_logging():
    registry = SkillRegistry()

    def my_skill(arg1):
        return f"Hello {arg1}"

    registry.register_skill("hello", my_skill, "Says hello")

    result = registry.execute_skill("hello", arg1="World")
    assert result == "Hello World"

    # Check audit trail
    logs = registry.acs_guard.audit_logger.get_all()
    # 1. intercept_action logs "All 5 checkpoints passed."
    # 2. SkillRegistry logs "Execution successful and sanitized."
    assert len(logs) == 2
    assert logs[0]["why"] == "All 5 checkpoints passed."
    assert logs[1]["why"] == "Execution successful and sanitized."
    assert logs[1]["result"] == "Hello World"
    assert logs[1]["duration"] >= 0

def test_skill_registry_audit_logging_error():
    registry = SkillRegistry()

    def failing_skill():
        raise ValueError("Boom")

    registry.register_skill("fail", failing_skill, "Always fails")

    # execute_skill catches internal exceptions and returns an error string
    result = registry.execute_skill("fail")
    assert "Error executing skill fail: Boom" in result

    logs = registry.acs_guard.audit_logger.get_all()
    assert len(logs) == 2
    assert logs[1]["why"] == "Execution error: Boom"
    assert logs[1]["result"] == "error"

def test_skill_registry_audit_logging_sanitization():
    registry = SkillRegistry()

    def secret_skill(password):
        return {"my_secret": password, "public": "ok"}

    registry.register_skill("secret", secret_skill, "Handles secrets")

    registry.execute_skill("secret", password="my_password")

    logs = registry.acs_guard.audit_logger.get_all()
    assert logs[0]["kwargs"]["password"] == "***"
    assert logs[1]["kwargs"]["password"] == "***"
    assert logs[1]["result"]["my_secret"] == "***"
    assert logs[1]["result"]["public"] == "ok"

import asyncio

@pytest.mark.asyncio
async def test_skill_registry_audit_logging_async():
    registry = SkillRegistry()

    async def my_async_skill(arg1):
        await asyncio.sleep(0.01)
        return f"Async {arg1}"

    registry.register_skill("hello_async", my_async_skill, "Says hello async")

    # execute_skill returns a coroutine because my_async_skill is a coroutine function
    coro = registry.execute_skill("hello_async", arg1="World")
    assert inspect.isawaitable(coro)

    result = await coro
    assert result == "Async World"

    logs = registry.acs_guard.audit_logger.get_all()
    assert len(logs) == 2
    assert logs[1]["why"] == "Execution successful and sanitized."
    assert logs[1]["result"] == "Async World"
    assert logs[1]["duration"] > 0

import inspect
