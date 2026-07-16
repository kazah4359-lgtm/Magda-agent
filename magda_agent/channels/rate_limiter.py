import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

class RateLimitExceededError(Exception):
    """Exception raised when a rate limit is exceeded and the strategy is set to block."""
    pass

class ChannelRateLimiter:
    """
    Rate limiter for outgoing channel messages.
    Supports 'delay' (holds the coroutine until a slot is free) and 'block' (raises RateLimitExceededError) strategies.
    """

    def __init__(
        self,
        limits: Optional[Dict[str, Dict[str, Any]]] = None,
        default_limit: Optional[Dict[str, Any]] = None,
        time_func: Optional[Callable[[], float]] = None,
        sleep_func: Optional[Callable[[float], Any]] = None
    ) -> None:
        """
        Initialize the rate limiter.

        Args:
            limits (Optional[Dict[str, Dict[str, Any]]]): Configuration dictionary per channel.
                Example: {"telegram": {"max_requests": 5, "period": 10.0, "strategy": "delay"}}
            default_limit (Optional[Dict[str, Any]]): Fallback limit config for unconfigured channels.
                Defaults to: {"max_requests": 5, "period": 1.0, "strategy": "delay"}
            time_func (Optional[Callable[[], float]]): Function returning current time as float.
            sleep_func (Optional[Callable[[float], Any]]): Async function to sleep/delay.
        """
        self.limits: Dict[str, Dict[str, Any]] = limits or {}
        self.default_limit: Dict[str, Any] = default_limit or {
            "max_requests": 5,
            "period": 1.0,
            "strategy": "delay"
        }
        self.time_func: Callable[[], float] = time_func or time.time
        self.sleep_func: Callable[[float], Any] = sleep_func or asyncio.sleep

        self._history: Dict[str, List[float]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_config(self, channel_id: str) -> Dict[str, Any]:
        """
        Retrieve rate limit configuration for the given channel, falling back to default.

        Args:
            channel_id (str): The channel ID.

        Returns:
            Dict[str, Any]: The rate limit configuration.
        """
        return self.limits.get(channel_id, self.default_limit)

    def _get_lock(self, channel_id: str) -> asyncio.Lock:
        """
        Retrieve or create an asyncio.Lock for the given channel to serialize requests.

        Args:
            channel_id (str): The channel ID.

        Returns:
            asyncio.Lock: The lock for the channel.
        """
        if channel_id not in self._locks:
            self._locks[channel_id] = asyncio.Lock()
        return self._locks[channel_id]

    async def acquire(self, channel_id: str) -> None:
        """
        Acquire a slot in the rate limit for the specified channel.
        Depending on the channel's strategy, this will either delay execution or raise RateLimitExceededError.

        Args:
            channel_id (str): The channel ID.

        Raises:
            RateLimitExceededError: If rate limit is exceeded and strategy is 'block'.
        """
        config = self._get_config(channel_id)
        max_requests: int = config.get("max_requests", 5)
        period: float = config.get("period", 1.0)
        strategy: str = config.get("strategy", "delay")

        lock = self._get_lock(channel_id)

        async with lock:
            if channel_id not in self._history:
                self._history[channel_id] = []

            now = self.time_func()
            # Clean up old timestamps outside the current sliding window
            self._history[channel_id] = [t for t in self._history[channel_id] if now - t < period]

            if len(self._history[channel_id]) < max_requests:
                self._history[channel_id].append(now)
            else:
                if strategy == "block":
                    raise RateLimitExceededError(
                        f"Rate limit exceeded on channel '{channel_id}'. "
                        f"Limit is {max_requests} requests per {period}s."
                    )
                elif strategy == "delay":
                    # Calculate the required delay time based on the oldest timestamp in the current window
                    earliest_timestamp = self._history[channel_id][0]
                    delay_time = (earliest_timestamp + period) - now
                    if delay_time > 0:
                        await self.sleep_func(delay_time)

                    # Update history after delay
                    now_after = self.time_func()
                    self._history[channel_id] = [t for t in self._history[channel_id] if now_after - t < period]
                    self._history[channel_id].append(now_after)
                else:
                    # Fallback to appending if strategy is unknown
                    self._history[channel_id].append(now)
