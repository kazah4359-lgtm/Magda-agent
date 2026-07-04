import asyncio
import logging
from dataclasses import dataclass
from typing import TypeVar, Callable, Awaitable, Any

T = TypeVar('T')

class A2ADelegationError(Exception):
    """Custom exception raised when an A2A delegation fails after all retries."""
    pass

@dataclass
class A2AErrorPolicy:
    """
    Defines the error handling and retry policy for A2A delegated tasks.
    """
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 10.0
    exponential_base: float = 2.0

class A2ARetryHandler:
    """
    Handles robust execution of A2A network operations with exponential backoff and retries.
    """
    def __init__(self, policy: A2AErrorPolicy | None = None):
        """
        Initializes the A2ARetryHandler with a given policy or a default one.

        Args:
            policy: Optional A2AErrorPolicy defining retry behaviors.
        """
        self.policy = policy or A2AErrorPolicy()
        self.logger = logging.getLogger("A2ARetryHandler")

    async def execute_with_retry(self, coro_func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """
        Executes an asynchronous callable with retry logic based on the configured policy.

        Args:
            coro_func: An asynchronous function to be executed.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the executed callable.

        Raises:
            A2ADelegationError: If the maximum number of retries is exceeded.
        """
        retries = 0
        delay = self.policy.base_delay_seconds

        while True:
            try:
                return await coro_func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries > self.policy.max_retries:
                    self.logger.error(f"Max retries ({self.policy.max_retries}) exceeded for A2A delegation. Final error: {e}")
                    raise A2ADelegationError(f"Delegation failed after {self.policy.max_retries} retries: {e}") from e

                self.logger.warning(f"A2A delegation failed (attempt {retries}/{self.policy.max_retries}): {e}. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
                delay = min(delay * self.policy.exponential_base, self.policy.max_delay_seconds)

def with_a2a_retry(policy: A2AErrorPolicy | None = None) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator for wrapping an async function with A2A retry logic.

    Args:
        policy: Optional A2AErrorPolicy to configure retry behavior.

    Returns:
        A decorated function that will automatically retry on failures.
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        handler = A2ARetryHandler(policy=policy)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await handler.execute_with_retry(func, *args, **kwargs)
        return wrapper
    return decorator
