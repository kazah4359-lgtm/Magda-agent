with open("tests/test_guardrails.py", "a") as f:
    f.write("""

def test_allowed_action_exception_fallback() -> None:
    \"\"\"Tests that an allowed action handles exceptions and returns a fallback message.\"\"\"
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(True, "Allowed"))
    guard = RealtimeGuardrail(policy)
    mock_tool = MagicMock(side_effect=ValueError("Test Error"))

    result = guard.execute_with_guardrails(mock_tool, "failing_tool", arg="value")
    assert result == "Action 'failing_tool' failed during execution: Test Error"
    mock_tool.assert_called_once_with(arg="value")

@pytest.mark.asyncio
async def test_allowed_action_async_exception_fallback() -> None:
    \"\"\"Tests that an allowed async action handles exceptions and returns a fallback message.\"\"\"
    policy = PolicyLayer()
    policy.evaluate = MagicMock(return_value=(True, "Allowed"))
    guard = RealtimeGuardrail(policy)

    async def failing_async_tool(arg: str) -> str:
        raise ValueError("Async Test Error")

    result_coro = guard.execute_with_guardrails(failing_async_tool, "failing_async_tool", arg="value")
    assert asyncio.iscoroutine(result_coro)
    result = await result_coro
    assert result == "Action 'failing_async_tool' failed during execution: Async Test Error"
""")
