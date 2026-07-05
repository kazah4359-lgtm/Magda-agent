import os
import sqlite3
import pytest
import pytest_asyncio
from magda_agent.operations.cron_backups import SQLiteCronBackupManager
from magda_agent.operations.cron_v3 import HermesCronSchedulerV3

from typing import List, Generator, AsyncGenerator
import pathlib

@pytest.fixture
def temp_dbs(tmp_path: pathlib.Path) -> List[str]:
    """
    Creates temporary SQLite databases and populates them with test data.

    Args:
        tmp_path: The temporary path fixture provided by pytest.

    Returns:
        A list of string paths to the temporary databases.
    """
    db1_path = tmp_path / "test1.db"
    db2_path = tmp_path / "test2.db"

    # Create dummy dbs
    conn1 = sqlite3.connect(db1_path)
    conn1.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn1.execute("INSERT INTO users (name) VALUES ('Alice')")
    conn1.commit()
    conn1.close()

    conn2 = sqlite3.connect(db2_path)
    conn2.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
    conn2.execute("INSERT INTO settings (key, value) VALUES ('theme', 'dark')")
    conn2.commit()
    conn2.close()

    return [str(db1_path), str(db2_path)]

@pytest.fixture
def backup_dir(tmp_path: pathlib.Path) -> str:
    """
    Creates a temporary directory for storing database backups.

    Args:
        tmp_path: The temporary path fixture provided by pytest.

    Returns:
        A string path to the created backup directory.
    """
    path = tmp_path / "backups"
    path.mkdir()
    return str(path)

@pytest_asyncio.fixture
async def backup_manager(temp_dbs: List[str], backup_dir: str) -> AsyncGenerator[SQLiteCronBackupManager, None]:
    """
    Initializes a SQLiteCronBackupManager with temporary databases and backup directory.

    Args:
        temp_dbs: The list of temporary database paths.
        backup_dir: The directory path for storing backups.

    Yields:
        An instance of SQLiteCronBackupManager ready for testing.
    """
    scheduler = HermesCronSchedulerV3()
    manager = SQLiteCronBackupManager(databases=temp_dbs, backup_dir=backup_dir, scheduler=scheduler)
    yield manager
    await manager.stop()

@pytest.mark.asyncio
async def test_backup_databases(backup_manager: SQLiteCronBackupManager, backup_dir: str) -> None:
    """
    Tests that the backup manager successfully copies SQLite databases to the backup directory.

    Args:
        backup_manager: The initialized SQLiteCronBackupManager fixture.
        backup_dir: The string path to the backup directory fixture.
    """
    await backup_manager.backup_databases()

    # Verify backup files were created
    files = os.listdir(backup_dir)
    assert len(files) == 2

    db1_backup = next(f for f in files if f.startswith("test1.db"))
    db2_backup = next(f for f in files if f.startswith("test2.db"))

    # Verify data in backups
    conn1 = sqlite3.connect(os.path.join(backup_dir, db1_backup))
    cursor1 = conn1.cursor()
    cursor1.execute("SELECT name FROM users")
    assert cursor1.fetchone()[0] == 'Alice'
    conn1.close()

    conn2 = sqlite3.connect(os.path.join(backup_dir, db2_backup))
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT value FROM settings")
    assert cursor2.fetchone()[0] == 'dark'
    conn2.close()

@pytest.mark.asyncio
async def test_register_nightly_backup(backup_manager: SQLiteCronBackupManager) -> None:
    """
    Tests that the nightly backup task is correctly registered with the cron scheduler.

    Args:
        backup_manager: The initialized SQLiteCronBackupManager fixture.
    """
    backup_manager.register_nightly_backup()

    # Verify the job is registered in the scheduler
    assert "sqlite_nightly_backup" in backup_manager.scheduler._func_registry
