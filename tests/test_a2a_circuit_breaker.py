import pytest
import asyncio
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

from magda_agent.integration.a2a_circuit_breaker import (
    A2ADelegationCircuitBreaker,
    CircuitState,
    CircuitBreakerOpenException,
)

@pytest.mark.asyncio
async def test_circuit_breaker_normal_execution() -> None:
    """
    Verifies that the circuit breaker allows execution when CLOSED and tracks state.
    """
    breaker = A2ADelegationCircuitBreaker(failure_threshold=2, recovery_timeout=0.1, success_threshold=2)
    peer_id = "peer-1"

    async def mock_delegate() -> Dict[str, Any]:
        return {"status": "success"}

    result = await breaker.execute(peer_id, mock_delegate)
    assert result == {"status": "success"}
    assert breaker.get_peer_state(peer_id) == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_trips_on_failures() -> None:
    """
    Verifies that the circuit breaker trips to OPEN after consecutive failures exceed threshold.
    """
    breaker = A2ADelegationCircuitBreaker(failure_threshold=2, recovery_timeout=1.0, success_threshold=2)
    peer_id = "peer-1"

    async def failing_delegate() -> Any:
        raise ValueError("Peer unresponsive")

    # First failure
    with pytest.raises(ValueError, match="Peer unresponsive"):
        await breaker.execute(peer_id, failing_delegate)
    assert breaker.get_peer_state(peer_id) == CircuitState.CLOSED

    # Second failure - should trip to OPEN
    with pytest.raises(ValueError, match="Peer unresponsive"):
        await breaker.execute(peer_id, failing_delegate)
    assert breaker.get_peer_state(peer_id) == CircuitState.OPEN

    # Execution is now blocked and raises CircuitBreakerOpenException
    with pytest.raises(CircuitBreakerOpenException):
        await breaker.execute(peer_id, failing_delegate)


@pytest.mark.asyncio
async def test_circuit_breaker_recovery_to_half_open_and_close() -> None:
    """
    Verifies that the circuit breaker recovers to HALF_OPEN after timeout and closes on successes.
    """
    breaker = A2ADelegationCircuitBreaker(failure_threshold=1, recovery_timeout=0.1, success_threshold=2)
    peer_id = "peer-1"

    async def failing_delegate() -> Any:
        raise ValueError("Peer unresponsive")

    # Trip the circuit
    with pytest.raises(ValueError):
        await breaker.execute(peer_id, failing_delegate)
    assert breaker.get_peer_state(peer_id) == CircuitState.OPEN

    # Wait for recovery timeout
    await asyncio.sleep(0.12)

    # Check state should become HALF_OPEN
    assert breaker.get_peer_state(peer_id) == CircuitState.HALF_OPEN

    # Successful execution trial 1
    async def successful_delegate() -> Dict[str, Any]:
        return {"result": "ok"}

    res = await breaker.execute(peer_id, successful_delegate)
    assert res == {"result": "ok"}
    assert breaker.get_peer_state(peer_id) == CircuitState.HALF_OPEN

    # Successful execution trial 2 - should close the circuit
    res2 = await breaker.execute(peer_id, successful_delegate)
    assert res2 == {"result": "ok"}
    assert breaker.get_peer_state(peer_id) == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_fails_fast() -> None:
    """
    Verifies that if an execution fails in HALF_OPEN, the circuit trips back to OPEN immediately.
    """
    breaker = A2ADelegationCircuitBreaker(failure_threshold=1, recovery_timeout=0.1, success_threshold=2)
    peer_id = "peer-1"

    async def failing_delegate() -> Any:
        raise ValueError("Peer unresponsive")

    # Trip the circuit
    with pytest.raises(ValueError):
        await breaker.execute(peer_id, failing_delegate)
    assert breaker.get_peer_state(peer_id) == CircuitState.OPEN

    # Wait for recovery timeout
    await asyncio.sleep(0.12)
    assert breaker.get_peer_state(peer_id) == CircuitState.HALF_OPEN

    # Failure in HALF_OPEN should trip back to OPEN immediately
    with pytest.raises(ValueError, match="Peer unresponsive"):
        await breaker.execute(peer_id, failing_delegate)
    assert breaker.get_peer_state(peer_id) == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_peer_isolation() -> None:
    """
    Verifies that circuit breaker state is isolated per peer.
    """
    breaker = A2ADelegationCircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
    peer_a = "peer-A"
    peer_b = "peer-B"

    async def failing_delegate() -> Any:
        raise ValueError("Failed")

    async def successful_delegate() -> str:
        return "success"

    # Trip Peer A
    with pytest.raises(ValueError):
        await breaker.execute(peer_a, failing_delegate)
    assert breaker.get_peer_state(peer_a) == CircuitState.OPEN

    # Peer B should still be CLOSED and functional
    assert breaker.get_peer_state(peer_b) == CircuitState.CLOSED
    res = await breaker.execute(peer_b, successful_delegate)
    assert res == "success"


@pytest.mark.asyncio
async def test_circuit_breaker_synchronous_fallback() -> None:
    """
    Verifies that synchronous fallback functions degrade execution gracefully.
    """
    breaker = A2ADelegationCircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
    peer_id = "peer-1"

    async def failing_delegate() -> Any:
        raise ValueError("Failed")

    def sync_fallback() -> Dict[str, Any]:
        return {"status": "fallback_local", "reason": "service_unavailable"}

    # Fails and triggers fallback
    res = await breaker.execute(peer_id, failing_delegate, fallback_func=sync_fallback)
    assert res == {"status": "fallback_local", "reason": "service_unavailable"}
    assert breaker.get_peer_state(peer_id) == CircuitState.OPEN

    # Blocked execution calls fallback directly
    res2 = await breaker.execute(peer_id, failing_delegate, fallback_func=sync_fallback)
    assert res2 == {"status": "fallback_local", "reason": "service_unavailable"}


@pytest.mark.asyncio
async def test_circuit_breaker_asynchronous_fallback() -> None:
    """
    Verifies that asynchronous fallback functions degrade execution gracefully.
    """
    breaker = A2ADelegationCircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
    peer_id = "peer-1"

    async def failing_delegate() -> Any:
        raise ValueError("Failed")

    async def async_fallback() -> Dict[str, Any]:
        return {"status": "fallback_async_local"}

    res = await breaker.execute(peer_id, failing_delegate, fallback_func=async_fallback)
    assert res == {"status": "fallback_async_local"}
