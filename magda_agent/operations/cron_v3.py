import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Callable, Any, Coroutine, Dict, List, Optional
from croniter import croniter

logger = logging.getLogger(__name__)


class HermesCronSchedulerV3:
    """
    A robust background job scheduler with persistence for autonomous tasks.
    Inspired by Hermes Agent cron scheduler for daily reports and nightly backups.
    Runs tasks periodically based on cron expressions without user interaction,
    persisting schedules to an SQLite database.
    """

    def __init__(self, db_path: str = ":memory:", result_callback: Optional[Callable[[Any], Coroutine[Any, Any, None]]] = None):
        """
        Initializes the HermesCronSchedulerV3.

        Args:
            db_path: Path to the SQLite database (defaults to in-memory).
            result_callback: Optional async callback to handle the result of a task execution.
        """
        # For memory, we want a shared memory instance if specified via URI,
        # but to keep it simple and testable without connections closing and dropping schema,
        # we will hold an internal connection open if it's an in-memory DB.
        self.db_path = db_path
        self._is_memory = db_path == ":memory:" or "mode=memory" in db_path

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.result_callback = result_callback

        # In-memory registry of functions
        self._func_registry: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}

        self._mem_conn: Optional[sqlite3.Connection] = None
        if self._is_memory:
            if "uri=True" in db_path or "?" in db_path:
                self._mem_conn = sqlite3.connect(self.db_path, uri=True)
            else:
                self._mem_conn = sqlite3.connect(self.db_path)

        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._mem_conn:
            return self._mem_conn
        if "?" in self.db_path:
            return sqlite3.connect(self.db_path, uri=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Initializes the SQLite database schema for jobs."""
        conn = self._get_conn()
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    name TEXT PRIMARY KEY,
                    cron_expr TEXT,
                    last_run TIMESTAMP,
                    next_run TIMESTAMP
                )
            ''')
            conn.commit()
        finally:
            if not self._is_memory:
                conn.close()

    def _get_now(self) -> datetime:
        """
        Returns the current UTC time as an aware datetime.
        Useful for mocking in tests.
        """
        return datetime.now(timezone.utc)

    def register_func(self, name: str, func: Callable[..., Coroutine[Any, Any, Any]]) -> None:
        """
        Registers an async function in the in-memory registry.

        Args:
            name: The unique name of the function/job.
            func: The async function.
        """
        self._func_registry[name] = func

    def schedule(self, cron_expr: str, func: Callable[..., Coroutine[Any, Any, Any]], name: Optional[str] = None) -> None:
        """
        Schedules a task to run according to a cron expression, persisting to DB.

        Args:
            cron_expr: The cron expression (e.g., "*/5 * * * *").
            func: The async function to execute.
            name: Optional name for the task. Defaults to function name.
        """
        if not croniter.is_valid(cron_expr):
            raise ValueError(f"Invalid cron expression: {cron_expr}")

        job_name = name or func.__name__
        self.register_func(job_name, func)

        now = self._get_now()
        itr = croniter(cron_expr, now)
        next_run = itr.get_next(datetime)

        conn = self._get_conn()
        try:
            conn.execute('''
                INSERT OR REPLACE INTO jobs (name, cron_expr, last_run, next_run)
                VALUES (?, ?, ?, ?)
            ''', (job_name, cron_expr, None, next_run.isoformat()))
            conn.commit()
        finally:
            if not self._is_memory:
                conn.close()

        logger.info(f"Scheduled generic task '{job_name}' with cron '{cron_expr}', next run: {next_run}")

    def task(self, cron_expr: str, name: Optional[str] = None) -> Callable[[Callable[..., Coroutine[Any, Any, Any]]], Callable[..., Coroutine[Any, Any, Any]]]:
        """
        A decorator to schedule a task via HermesCronSchedulerV3.

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

    async def start(self) -> None:
        """
        Starts the generic scheduler loop in the background.
        """
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("HermesCronSchedulerV3 started.")

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

        if self._mem_conn:
            self._mem_conn.close()
            self._mem_conn = None

        logger.info("HermesCronSchedulerV3 stopped.")

    async def _loop(self) -> None:
        """
        The main loop checking for jobs to run based on DB state.
        """
        while self._running:
            now = self._get_now()
            jobs_to_run = []

            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cursor.execute('SELECT name, cron_expr, next_run FROM jobs')
                rows = cursor.fetchall()

                for row in rows:
                    name, cron_expr, next_run_str = row
                    next_run = datetime.fromisoformat(next_run_str)

                    if now >= next_run:
                        # Ensure timezone awareness matches
                        if next_run.tzinfo is None:
                            next_run = next_run.replace(tzinfo=timezone.utc)

                        jobs_to_run.append({
                            "name": name,
                            "cron_expr": cron_expr,
                            "next_run": next_run
                        })
            finally:
                if not self._is_memory:
                    conn.close()

            for job in jobs_to_run:
                asyncio.create_task(self._execute_job(job))

            await asyncio.sleep(1.0) # Check every second

    async def _execute_job(self, job: Dict[str, Any]) -> None:
        """
        Executes a scheduled job, updates next_run in DB, and optionally calls the result callback.

        Args:
            job: The job dictionary containing execution details from DB.
        """
        name = job["name"]
        cron_expr = job["cron_expr"]

        func = self._func_registry.get(name)
        if not func:
            logger.error(f"Function for job '{name}' is not registered.")
            return

        now = self._get_now()

        # Calculate next run
        itr = croniter(cron_expr, now)
        next_run = itr.get_next(datetime)

        # Update DB before execution to prevent double execution if long running
        conn = self._get_conn()
        try:
            conn.execute('''
                UPDATE jobs SET last_run = ?, next_run = ? WHERE name = ?
            ''', (now.isoformat(), next_run.isoformat(), name))
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating job {name} in DB: {e}")
            return
        finally:
            if not self._is_memory:
                conn.close()

        try:
            logger.info(f"Executing scheduled generic task: {name}")
            result = await func()
            if self.result_callback and result is not None:
                await self.result_callback(result)
        except Exception as e:
            logger.error(f"Error executing generic scheduled task {name}: {e}", exc_info=True)
