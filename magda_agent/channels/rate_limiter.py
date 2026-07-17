import asyncio
import time
from typing import Dict, Any, Tuple, Callable, Optional

class RateLimitExceeded(Exception):
    """Exception raised when the rate limit is exceeded in block mode."""
    pass

class ChannelRateLimiter:
    """
    Token-bucket based rate limiter for communication channels.
    Supports both delay mode (throttling/sleeping) and block mode (raising RateLimitExceeded).
    """
    def __init__(
        self,
        default_rate_limit: int = 5,
        default_window: float = 10.0,
        block_mode: bool = False,
        channel_limits: Optional[Dict[str, Tuple[int, float]]] = None,
        time_func: Callable[[], float] = time.time,
        sleep_func: Callable[[float], Any] = asyncio.sleep,
    ):
        """
        Args:
            default_rate_limit: Default max tokens.
            default_window: Default window duration in seconds.
            block_mode: If True, raises RateLimitExceeded on violation. If False, delays.
            channel_limits: Dict mapping channel_id -> (rate_limit, window).
            time_func: Function to get the current time.
            sleep_func: Async function to perform sleep/delay.
        """
        self.default_rate_limit = default_rate_limit
        self.default_window = default_window
        self.block_mode = block_mode
        self.channel_limits = channel_limits or {}
        self.time_func = time_func
        self.sleep_func = sleep_func

        # State storage: channel_id -> { "tokens": float, "last_updated": float }
        self._states: Dict[str, Dict[str, float]] = {}

    def _get_limit_and_refill_rate(self, channel_id: str) -> Tuple[float, float]:
        """Returns (max_tokens, refill_rate) for a given channel."""
        limit, window = self.channel_limits.get(channel_id, (self.default_rate_limit, self.default_window))
        max_tokens = float(limit)
        refill_rate = max_tokens / float(window) if window > 0 else float('inf')
        return max_tokens, refill_rate

    def _init_state_if_needed(self, channel_id: str, max_tokens: float, now: float) -> None:
        if channel_id not in self._states:
            self._states[channel_id] = {
                "tokens": max_tokens,
                "last_updated": now
            }

    async def acquire(self, channel_id: str) -> None:
        """
        Attempt to acquire a token for the given channel_id.
        Delays or blocks depending on block_mode.
        """
        max_tokens, refill_rate = self._get_limit_and_refill_rate(channel_id)
        if refill_rate == float('inf'):
            return

        now = self.time_func()
        self._init_state_if_needed(channel_id, max_tokens, now)

        state = self._states[channel_id]
        elapsed = max(0.0, now - state["last_updated"])

        # Refill tokens based on elapsed time
        refilled_tokens = state["tokens"] + (elapsed * refill_rate)
        state["tokens"] = min(max_tokens, refilled_tokens)
        state["last_updated"] = now

        if state["tokens"] >= 1.0:
            # Token available immediately
            state["tokens"] -= 1.0
        else:
            # Token not available
            needed = 1.0 - state["tokens"]
            wait_time = needed / refill_rate

            if self.block_mode:
                raise RateLimitExceeded(
                    f"Rate limit exceeded for channel '{channel_id}'. "
                    f"Needs {wait_time:.2f}s to recover."
                )
            else:
                # Delay mode: wait for token to become available
                await self.sleep_func(wait_time)
                # After sleeping, update our tokens state as if wait_time has elapsed
                state["tokens"] = 0.0
                state["last_updated"] = now + wait_time
