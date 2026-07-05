import os
import sqlite3
import pytest
import pytest_asyncio
from magda_agent.operations.cron_backups_v2 import SQLiteCronBackupManagerV2
from magda_agent.operations.cron_v3 import HermesCronSchedulerV3

from typing import List, AsyncGenerator
import pathlib

@pytest.fixture
def temp_dbs_v2(tmp_path: pathlib.Path) -> List[str]:
    """
    Creates temporary SQLite databases and populates them with test data.

    Args:
        tmp_path: The temporary path fixture provided by pytest.

    Returns:
        A list of string paths to the temporary databases.
    """
    db1_path = tmp_path / "test1_v2.db"
    db2_path = tmp_path / "test2_v2.db"

    # Create dummy dbs
    conn1 = sqlite3.connect(db1_path)
    conn1.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn1.execute("INSERT INTO users (name) VALUES ('Bob')")
    conn1.commit()
    conn1.close()

    conn2 = sqlite3.connect(db2_path)
    conn2.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
    conn2.execute("INSERT INTO settings (key, value) VALUES ('theme', 'light')")
    conn2.commit()
    conn2.close()

    return [str(db1_path), str(db2_path)]

@pytest.fixture
def backup_dir_v2(tmp_path: pathlib.Path) -> str:
    """
    Creates a temporary directory for storing database backups.

    Args:
        tmp_path: The temporary path fixture provided by pytest.

    Returns:
        A string path to the created backup directory.
    """
    path = tmp_path / "backups_v2"
    path.mkdir()
    return str(path)

@pytest_asyncio.fixture
async def backup_manager_v2(temp_dbs_v2: List[str], backup_dir_v2: str) -> AsyncGenerator[SQLiteCronBackupManagerV2, None]:
    """
    Initializes a SQLiteCronBackupManagerV2 with temporary databases and backup directory.

    Args:
        temp_dbs_v2: The list of temporary database paths.
        backup_dir_v2: The directory path for storing backups.

    Yields:
        An instance of SQLiteCronBackupManagerV2 ready for testing.
    """
    scheduler = HermesCronSchedulerV3()
    manager = SQLiteCronBackupManagerV2(databases=temp_dbs_v2, backup_dir=backup_dir_v2, scheduler=scheduler)
    yield manager
    await manager.stop()

@pytest.mark.asyncio
async def test_backup_databases_v2(backup_manager_v2: SQLiteCronBackupManagerV2, backup_dir_v2: str) -> None:
    """
    Tests that the backup manager successfully copies SQLite databases to the backup directory.

    Args:
        backup_manager_v2: The initialized SQLiteCronBackupManagerV2 fixture.
        backup_dir_v2: The string path to the backup directory fixture.
    """
    await backup_manager_v2.backup_databases()

    # Verify backup files were created
    files = os.listdir(backup_dir_v2)
    assert len(files) == 2

    db1_backup = next(f for f in files if f.startswith("test1_v2.db"))
    db2_backup = next(f for f in files if f.startswith("test2_v2.db"))

    # Verify data in backups
    conn1 = sqlite3.connect(os.path.join(backup_dir_v2, db1_backup))
    cursor1 = conn1.cursor()
    cursor1.execute("SELECT name FROM users")
    assert cursor1.fetchone()[0] == 'Bob'
    conn1.close()

    conn2 = sqlite3.connect(os.path.join(backup_dir_v2, db2_backup))
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT value FROM settings")
    assert cursor2.fetchone()[0] == 'light'
    conn2.close()

@pytest.mark.asyncio
async def test_register_nightly_backup_v2(backup_manager_v2: SQLiteCronBackupManagerV2) -> None:
    """
    Tests that the nightly backup task is correctly registered with the cron scheduler.

    Args:
        backup_manager_v2: The initialized SQLiteCronBackupManagerV2 fixture.
    """
    backup_manager_v2.register_nightly_backup()

    # Verify the job is registered in the scheduler
    assert "sqlite_nightly_backup_v2" in backup_manager_v2.scheduler._func_registry
