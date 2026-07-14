from enum import Enum
import time
import logging
from typing import Any, Dict, Callable, Optional, Coroutine

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """
    Represents the states of the circuit breaker.
    """
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreakerOpenException(Exception):
    """
    Exception raised when a request is blocked because the circuit breaker is OPEN.
    """
    pass

class PeerState:
    """
    Represents the state of a single peer agent.
    """
    def __init__(self) -> None:
        """
        Initializes the state for a single peer agent.
        """
        self.state: CircuitState = CircuitState.CLOSED
        self.consecutive_failures: int = 0
        self.consecutive_successes: int = 0
        self.last_state_change: float = time.time()

class A2ADelegationCircuitBreaker:
    """
    A circuit breaker implementation for A2A (Agent-to-Agent) task delegation.
    Prevents continuous task delegation failures to unresponsive peer agents
    by tripping the circuit and gracefully degrading.
    """
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 5.0,
        success_threshold: int = 2
    ) -> None:
        """
        Initializes the circuit breaker with custom thresholds.

        Args:
            failure_threshold (int): Number of consecutive failures to trip the circuit.
            recovery_timeout (float): Time in seconds to wait before checking if peer recovered.
            success_threshold (int): Number of consecutive successes in HALF_OPEN to close the circuit.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.peer_states: Dict[str, PeerState] = {}

    def _get_or_create_peer_state(self, peer_id: str) -> PeerState:
        """
        Retrieves or initializes the state tracker for a given peer agent ID.
        """
        if peer_id not in self.peer_states:
            self.peer_states[peer_id] = PeerState()
        return self.peer_states[peer_id]

    def get_peer_state(self, peer_id: str) -> CircuitState:
        """
        Gets the current state of the circuit breaker for a peer agent.

        Args:
            peer_id (str): The identifier of the peer agent.

        Returns:
            CircuitState: The current circuit state (CLOSED, OPEN, HALF_OPEN).
        """
        peer_state = self._get_or_create_peer_state(peer_id)
        if peer_state.state == CircuitState.OPEN:
            now = time.time()
            if now - peer_state.last_state_change >= self.recovery_timeout:
                # Transition to HALF_OPEN
                logger.info(f"Circuit breaker for peer {peer_id} transitioning from OPEN to HALF_OPEN due to timeout.")
                peer_state.state = CircuitState.HALF_OPEN
                peer_state.consecutive_successes = 0
                peer_state.last_state_change = now
        return peer_state.state

    def is_execution_allowed(self, peer_id: str) -> bool:
        """
        Determines whether execution is allowed for a peer agent.

        Args:
            peer_id (str): The identifier of the peer agent.

        Returns:
            bool: True if execution is allowed, False otherwise.
        """
        state = self.get_peer_state(peer_id)
        return state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self, peer_id: str) -> None:
        """
        Records a successful execution, updating the peer's state.

        Args:
            peer_id (str): The identifier of the peer agent.
        """
        peer_state = self._get_or_create_peer_state(peer_id)
        if peer_state.state == CircuitState.HALF_OPEN:
            peer_state.consecutive_successes += 1
            if peer_state.consecutive_successes >= self.success_threshold:
                logger.info(f"Circuit breaker for peer {peer_id} reset to CLOSED after {peer_state.consecutive_successes} successes.")
                peer_state.state = CircuitState.CLOSED
                peer_state.consecutive_failures = 0
                peer_state.consecutive_successes = 0
                peer_state.last_state_change = time.time()
        elif peer_state.state == CircuitState.CLOSED:
            peer_state.consecutive_failures = 0

    def record_failure(self, peer_id: str) -> None:
        """
        Records a failed execution, possibly tripping the circuit.

        Args:
            peer_id (str): The identifier of the peer agent.
        """
        peer_state = self._get_or_create_peer_state(peer_id)
        now = time.time()
        if peer_state.state == CircuitState.CLOSED:
            peer_state.consecutive_failures += 1
            if peer_state.consecutive_failures >= self.failure_threshold:
                logger.warning(f"Circuit breaker for peer {peer_id} tripped to OPEN after {peer_state.consecutive_failures} failures.")
                peer_state.state = CircuitState.OPEN
                peer_state.last_state_change = now
        elif peer_state.state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit breaker for peer {peer_id} tripped back to OPEN after failure in HALF_OPEN.")
            peer_state.state = CircuitState.OPEN
            peer_state.consecutive_failures = self.failure_threshold
            peer_state.consecutive_successes = 0
            peer_state.last_state_change = now

    async def execute(
        self,
        peer_id: str,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        fallback_func: Optional[Callable[..., Any]] = None,
        **kwargs: Any
    ) -> Any:
        """
        Executes an asynchronous function with circuit breaker protection.

        Args:
            peer_id (str): The identifier of the peer agent.
            func (Callable): The asynchronous function/coroutine to execute.
            args (Any): Variable positional arguments to pass to the function.
            fallback_func (Optional[Callable]): Optional sync or async function to call on failure/block.
            kwargs (Any): Variable keyword arguments to pass to the function.

        Returns:
            Any: The result of the function execution, or fallback.

        Raises:
            CircuitBreakerOpenException: If the circuit is OPEN and no fallback_func is provided.
        """
        if not self.is_execution_allowed(peer_id):
            logger.warning(f"Execution blocked by circuit breaker for peer {peer_id}.")
            if fallback_func is not None:
                return await self._execute_fallback(fallback_func, *args, **kwargs)
            raise CircuitBreakerOpenException(f"Circuit breaker for peer {peer_id} is OPEN.")

        try:
            result = await func(*args, **kwargs)
            self.record_success(peer_id)
            return result
        except Exception as e:
            logger.exception(f"Exception during execution under circuit breaker for peer {peer_id}: {e}")
            self.record_failure(peer_id)
            if fallback_func is not None:
                return await self._execute_fallback(fallback_func, *args, **kwargs)
            raise

    async def _execute_fallback(self, fallback_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Helper to execute sync or async fallbacks.
        """
        import inspect
        if inspect.iscoroutinefunction(fallback_func):
            return await fallback_func(*args, **kwargs)
        else:
            return fallback_func(*args, **kwargs)
