import logging
import sqlite3
import os
import asyncio
from datetime import datetime, timezone
from typing import Optional, List

from magda_agent.operations.cron_v3 import HermesCronSchedulerV3

logger = logging.getLogger(__name__)

class SQLiteCronBackupManagerV3:
    """
    Manages the scheduled backup operations for SQLite databases (such as episodic memory dumps)
    using HermesCronSchedulerV3.
    """

    def __init__(
        self,
        databases: List[str],
        backup_dir: str,
        scheduler: Optional[HermesCronSchedulerV3] = None
    ):
        """
        Initializes the SQLiteCronBackupManagerV3.

        Args:
            databases: A list of paths to the SQLite databases to backup.
            backup_dir: The directory where backups will be stored.
            scheduler: The HermesCronSchedulerV3 instance. If None, a new one is created.
        """
        self.databases = databases
        self.backup_dir = backup_dir
        self.scheduler = scheduler or HermesCronSchedulerV3()

    async def backup_databases(self) -> None:
        """
        Performs the backup operation for all registered databases.
        Copies the databases to the backup directory using sqlite3 backup API.
        """
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir, exist_ok=True)

        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        for db_path in self.databases:
            if not os.path.exists(db_path):
                logger.warning(f"Database {db_path} does not exist, skipping backup.")
                continue

            db_name = os.path.basename(db_path)
            backup_path = os.path.join(self.backup_dir, f"{db_name}.{now_str}.bak")

            try:
                def _do_backup() -> None:
                    src_conn = sqlite3.connect(db_path)
                    dst_conn = sqlite3.connect(backup_path)
                    try:
                        with src_conn, dst_conn:
                            src_conn.backup(dst_conn)
                    finally:
                        src_conn.close()
                        dst_conn.close()

                await asyncio.to_thread(_do_backup)

                logger.info(f"Successfully backed up episodic memory DB {db_path} to {backup_path}")
            except Exception as e:
                logger.error(f"Failed to backup {db_path} to {backup_path}: {e}")

    def register_nightly_backup(self, cron_expr: str = "0 2 * * *", name: str = "sqlite_nightly_backup_v3") -> None:
        """
        Registers the backup operation as a nightly task.

        Args:
            cron_expr: The cron expression indicating when to run the backup. Defaults to "0 2 * * *" (2:00 AM daily).
            name: The task name registered in the scheduler.
        """
        self.scheduler.schedule(cron_expr, self.backup_databases, name=name)
        logger.info(f"Registered nightly backup task '{name}' with schedule '{cron_expr}'")

    async def start(self) -> None:
        """
        Starts the underlying scheduler loop.
        """
        logger.info("Starting SQLiteCronBackupManagerV3 scheduler")
        await self.scheduler.start()

    async def stop(self) -> None:
        """
        Stops the underlying scheduler loop.
        """
        logger.info("Stopping SQLiteCronBackupManagerV3 scheduler")
        await self.scheduler.stop()


class HermesEpisodicMemoryBackupManagerV3(SQLiteCronBackupManagerV3):
    """
    Convenience wrapper specifically designed for Hermes Agent episodic memory dumps.
    """

    def __init__(
        self,
        memory_dbs: Optional[List[str]] = None,
        backup_dir: str = "backups/episodic_memory",
        scheduler: Optional[HermesCronSchedulerV3] = None
    ):
        dbs = memory_dbs if memory_dbs is not None else ["episodic_memory.db"]
        super().__init__(databases=dbs, backup_dir=backup_dir, scheduler=scheduler)

    def schedule_nightly_episodic_dump(self, cron_expr: str = "0 2 * * *") -> None:
        """
        Schedules a nightly backup task for episodic memory dumps.
        """
        self.register_nightly_backup(cron_expr=cron_expr, name="hermes_nightly_episodic_memory_dump_v3")
