import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from magda_agent.integration.a2a_error import A2AErrorPolicy, A2ARetryHandler, A2ADelegationError, with_a2a_retry

@pytest.mark.asyncio
async def test_successful_execution():
    """Test that execution succeeds on the first try without retries."""
    mock_coro = AsyncMock(return_value="success")
    handler = A2ARetryHandler()
    result = await handler.execute_with_retry(mock_coro, "arg1", kwarg1="val1")

    assert result == "success"
    mock_coro.assert_called_once_with("arg1", kwarg1="val1")

@pytest.mark.asyncio
async def test_retry_on_failure_then_success():
    """Test that execution retries on failure and succeeds eventually."""
    mock_coro = AsyncMock(side_effect=[Exception("network error"), Exception("timeout"), "success"])
    policy = A2AErrorPolicy(max_retries=3, base_delay_seconds=0.01)
    handler = A2ARetryHandler(policy=policy)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await handler.execute_with_retry(mock_coro)

        assert result == "success"
        assert mock_coro.call_count == 3
        assert mock_sleep.call_count == 2
        # Check that delay increased exponentially
        assert mock_sleep.call_args_list[0][0][0] == 0.01
        assert mock_sleep.call_args_list[1][0][0] == 0.02

@pytest.mark.asyncio
async def test_max_retries_exceeded():
    """Test that A2ADelegationError is raised when max retries are exceeded."""
    mock_coro = AsyncMock(side_effect=Exception("persistent error"))
    policy = A2AErrorPolicy(max_retries=2, base_delay_seconds=0.01)
    handler = A2ARetryHandler(policy=policy)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(A2ADelegationError, match="Delegation failed after 2 retries"):
            await handler.execute_with_retry(mock_coro)

        assert mock_coro.call_count == 3  # 1 initial try + 2 retries

@pytest.mark.asyncio
async def test_decorator_usage():
    """Test that the with_a2a_retry decorator correctly applies retry logic."""
    policy = A2AErrorPolicy(max_retries=1, base_delay_seconds=0.01)

    mock_coro = AsyncMock(side_effect=[Exception("temp fail"), "decorated_success"])

    @with_a2a_retry(policy=policy)
    async def my_decorated_func(*args, **kwargs):
        return await mock_coro(*args, **kwargs)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await my_decorated_func(1, 2, c=3)

    assert result == "decorated_success"
    assert mock_coro.call_count == 2
    mock_coro.assert_called_with(1, 2, c=3)
