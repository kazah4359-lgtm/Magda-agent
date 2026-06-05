import asyncio
import logging
from datetime import datetime
from typing import Callable, Any, Coroutine, Dict, List
from croniter import croniter

logger = logging.getLogger(__name__)

class CronScheduler:
    """
    A lightweight cron-like scheduler for autonomous tasks.
    Runs tasks periodically based on cron expressions.
    """

    def __init__(self, result_callback: Callable[[Any], Coroutine[Any, Any, None]] = None):
        """
        Initializes the CronScheduler.

        Args:
            result_callback: Optional async callback to handle the result of a task execution.
                             Often used to deliver results to a configured channel.
        """
        self.jobs: List[Dict[str, Any]] = []
        self._running = False
        self._task: asyncio.Task = None
        self.result_callback = result_callback

    def schedule(self, cron_expr: str, func: Callable[..., Coroutine[Any, Any, Any]], *args, **kwargs) -> None:
        """
        Schedules a task to run according to a cron expression.

        Args:
            cron_expr: The cron expression (e.g., "*/5 * * * *").
            func: The async function to execute.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        """
        if not croniter.is_valid(cron_expr):
            raise ValueError(f"Invalid cron expression: {cron_expr}")

        now = self._get_now()
        itr = croniter(cron_expr, now)
        next_run = itr.get_next(datetime)

        job = {
            "cron_expr": cron_expr,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "next_run": next_run,
            "iterator": itr
        }
        self.jobs.append(job)
        logger.info(f"Scheduled task {func.__name__} with cron '{cron_expr}', next run: {next_run}")

    def _get_now(self) -> datetime:
        """
        Returns the current time. Useful for mocking in tests.
        """
        return datetime.now()

    async def start(self) -> None:
        """
        Starts the scheduler loop in the background.
        """
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("CronScheduler started.")

    async def stop(self) -> None:
        """
        Stops the scheduler loop.
        """
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("CronScheduler stopped.")

    async def _loop(self) -> None:
        """
        The main loop checking for jobs to run.
        """
        while self._running:
            now = self._get_now()
            for job in self.jobs:
                if now >= job["next_run"]:
                    # Execute task
                    asyncio.create_task(self._execute_job(job))
                    # Update next run time
                    job["next_run"] = job["iterator"].get_next(datetime)
            await asyncio.sleep(1.0) # Check every second

    async def _execute_job(self, job: Dict[str, Any]) -> None:
        """
        Executes a scheduled job and optionally calls the result callback.
        """
        func = job["func"]
        try:
            logger.info(f"Executing scheduled task: {func.__name__}")
            result = await func(*job["args"], **job["kwargs"])
            if self.result_callback and result is not None:
                await self.result_callback(result)
        except Exception as e:
            logger.error(f"Error executing scheduled task {func.__name__}: {e}", exc_info=True)
