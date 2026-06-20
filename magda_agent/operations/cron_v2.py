import asyncio
import logging
from datetime import datetime
from typing import Callable, Any, Coroutine, Dict, List, Optional
from croniter import croniter

logger = logging.getLogger(__name__)

class GenericCronSchedulerV2:
    """
    A generic cron-like scheduler for autonomous tasks based on trend: Scheduled operations.
    Runs tasks periodically based on cron expressions without user interaction.
    """

    def __init__(self, result_callback: Optional[Callable[[Any], Coroutine[Any, Any, None]]] = None):
        """
        Initializes the GenericCronSchedulerV2.

        Args:
            result_callback: Optional async callback to handle the result of a task execution.
        """
        self.jobs: List[Dict[str, Any]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.result_callback = result_callback

    def schedule(self, cron_expr: str, func: Callable[..., Coroutine[Any, Any, Any]], name: Optional[str] = None, *args, **kwargs) -> None:
        """
        Schedules a task to run according to a cron expression.

        Args:
            cron_expr: The cron expression (e.g., "*/5 * * * *").
            func: The async function to execute.
            name: Optional name for the task.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        """
        if not croniter.is_valid(cron_expr):
            raise ValueError(f"Invalid cron expression: {cron_expr}")

        now = self._get_now()
        itr = croniter(cron_expr, now)
        next_run = itr.get_next(datetime)

        job_name = name or func.__name__

        job = {
            "name": job_name,
            "cron_expr": cron_expr,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "next_run": next_run,
            "iterator": itr
        }
        self.jobs.append(job)
        logger.info(f"Scheduled generic task '{job_name}' with cron '{cron_expr}', next run: {next_run}")

    def task(self, cron_expr: str, name: Optional[str] = None) -> Callable[[Callable[..., Coroutine[Any, Any, Any]]], Callable[..., Coroutine[Any, Any, Any]]]:
        """
        A decorator to schedule a task via GenericCronSchedulerV2.

        Args:
            cron_expr: The cron expression.
            name: Optional name for the task.

        Returns:
            The decorator function.
        """
        def decorator(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
            self.schedule(cron_expr, func, name=name)
            return func
        return decorator

    def _get_now(self) -> datetime:
        """
        Returns the current time. Useful for mocking in tests.
        """
        return datetime.now()

    async def start(self) -> None:
        """
        Starts the generic scheduler loop in the background.
        """
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("GenericCronSchedulerV2 started.")

    async def stop(self) -> None:
        """
        Stops the generic scheduler loop.
        """
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("GenericCronSchedulerV2 stopped.")

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

        Args:
            job: The job dictionary containing execution details.
        """
        func = job["func"]
        name = job["name"]
        try:
            logger.info(f"Executing scheduled generic task: {name}")
            result = await func(*job["args"], **job["kwargs"])
            if self.result_callback and result is not None:
                await self.result_callback(result)
        except Exception as e:
            logger.error(f"Error executing generic scheduled task {name}: {e}", exc_info=True)
