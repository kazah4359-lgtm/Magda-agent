import asyncio
import logging
from datetime import datetime
from typing import Callable, Any, Coroutine, Dict, List
from croniter import croniter

logger = logging.getLogger(__name__)

class OperationsCronScheduler:
    """
    Operations-specific lightweight cron-like scheduler for autonomous tasks.
    Runs tasks periodically based on cron expressions.
    """

    def __init__(self, result_callback: Callable[[Any], Coroutine[Any, Any, None]] = None) -> None:
        """
        Initializes the OperationsCronScheduler.

        Args:
            result_callback: Optional async callback to handle the result of a task execution.
        """
        self.jobs: List[Dict[str, Any]] = []
        self._running: bool = False
        self._task: asyncio.Task | None = None
        self.result_callback = result_callback

    def schedule(self, cron_expr: str, func: Callable[..., Coroutine[Any, Any, Any]], name: str | None = None, *args, **kwargs) -> None:
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
        logger.info(f"Scheduled operations task '{job_name}' with cron '{cron_expr}', next run: {next_run}")

    def task(self, cron_expr: str, name: str | None = None) -> Callable[[Callable[..., Coroutine[Any, Any, Any]]], Callable[..., Coroutine[Any, Any, Any]]]:
        """
        A decorator to schedule a task according to a cron expression.

        Args:
            cron_expr: The cron expression (e.g., "*/5 * * * *").
            name: Optional name for the task.
        """
        def decorator(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
            """
            Internal decorator function.
            """
            self.schedule(cron_expr, func, name=name)
            return func
        return decorator

    def _get_now(self) -> datetime:
        """
        Returns the current time. Useful for mocking in tests.

        Returns:
            The current datetime.
        """
        return datetime.now()

    async def start(self) -> None:
        """
        Starts the operations scheduler loop in the background.
        """
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("OperationsCronScheduler started.")

    async def stop(self) -> None:
        """
        Stops the operations scheduler loop.
        """
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("OperationsCronScheduler stopped.")

    async def _loop(self) -> None:
        """
        The main loop checking for jobs to run.
        """
        while self._running:
            now = self._get_now()
            for job in self.jobs:
                if now >= job["next_run"]:
                    asyncio.create_task(self._execute_job(job))
                    job["next_run"] = job["iterator"].get_next(datetime)
            await asyncio.sleep(1.0)

    async def _execute_job(self, job: Dict[str, Any]) -> None:
        """
        Executes a scheduled job and optionally calls the result callback.

        Args:
            job: The dictionary containing the job execution details.
        """
        func = job["func"]
        name = job["name"]
        try:
            logger.info(f"Executing scheduled operations task: {name}")
            result = await func(*job["args"], **job["kwargs"])
            if self.result_callback and result is not None:
                await self.result_callback(result)
        except Exception as e:
            logger.error(f"Error executing scheduled operations task {name}: {e}", exc_info=True)
