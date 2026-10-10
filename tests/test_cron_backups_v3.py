import os
import sqlite3
import pytest
import pytest_asyncio
import pathlib
from typing import List, AsyncGenerator

from magda_agent.operations.cron_backups_v3 import (
    SQLiteCronBackupManagerV3,
    HermesEpisodicMemoryBackupManagerV3,
)
from magda_agent.operations.cron_v3 import HermesCronSchedulerV3


@pytest.fixture
def temp_episodic_dbs_v3(tmp_path: pathlib.Path) -> List[str]:
    """
    Creates temporary SQLite episodic memory databases with test records.
    """
    db1_path = tmp_path / "episodic_memory_1.db"
    db2_path = tmp_path / "episodic_memory_2.db"

    conn1 = sqlite3.connect(db1_path)
    conn1.execute("CREATE TABLE memory_entries (id INTEGER PRIMARY KEY, content TEXT)")
    conn1.execute("INSERT INTO memory_entries (content) VALUES ('User prefers dark mode')")
    conn1.commit()
    conn1.close()

    conn2 = sqlite3.connect(db2_path)
    conn2.execute("CREATE TABLE episodic_logs (id INTEGER PRIMARY KEY, summary TEXT)")
    conn2.execute("INSERT INTO episodic_logs (summary) VALUES ('Completed task successfully')")
    conn2.commit()
    conn2.close()

    return [str(db1_path), str(db2_path)]


@pytest.fixture
def backup_dir_v3(tmp_path: pathlib.Path) -> str:
    """
    Creates a temporary directory for storing database backups.
    """
    path = tmp_path / "backups_v3"
    path.mkdir()
    return str(path)


@pytest_asyncio.fixture
async def backup_manager_v3(
    temp_episodic_dbs_v3: List[str], backup_dir_v3: str
) -> AsyncGenerator[SQLiteCronBackupManagerV3, None]:
    """
    Initializes a SQLiteCronBackupManagerV3 with temporary databases and backup directory.
    """
    scheduler = HermesCronSchedulerV3()
    manager = SQLiteCronBackupManagerV3(
        databases=temp_episodic_dbs_v3, backup_dir=backup_dir_v3, scheduler=scheduler
    )
    yield manager
    await manager.stop()


@pytest.mark.asyncio
async def test_backup_databases_v3(
    backup_manager_v3: SQLiteCronBackupManagerV3, backup_dir_v3: str
) -> None:
    """
    Tests that backup_databases successfully creates timestamped backup files with valid data.
    """
    await backup_manager_v3.backup_databases()

    files = os.listdir(backup_dir_v3)
    assert len(files) == 2

    db1_backup = next(f for f in files if f.startswith("episodic_memory_1.db"))
    db2_backup = next(f for f in files if f.startswith("episodic_memory_2.db"))

    conn1 = sqlite3.connect(os.path.join(backup_dir_v3, db1_backup))
    cursor1 = conn1.cursor()
    cursor1.execute("SELECT content FROM memory_entries")
    assert cursor1.fetchone()[0] == "User prefers dark mode"
    conn1.close()

    conn2 = sqlite3.connect(os.path.join(backup_dir_v3, db2_backup))
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT summary FROM episodic_logs")
    assert cursor2.fetchone()[0] == "Completed task successfully"
    conn2.close()


@pytest.mark.asyncio
async def test_register_nightly_backup_v3(
    backup_manager_v3: SQLiteCronBackupManagerV3,
) -> None:
    """
    Tests that the nightly backup task registers correctly with HermesCronSchedulerV3.
    """
    backup_manager_v3.register_nightly_backup()

    assert "sqlite_nightly_backup_v3" in backup_manager_v3.scheduler._func_registry


@pytest.mark.asyncio
async def test_hermes_episodic_memory_backup_manager_v3(
    temp_episodic_dbs_v3: List[str], backup_dir_v3: str
) -> None:
    """
    Tests HermesEpisodicMemoryBackupManagerV3 convenience class scheduling and execution.
    """
    scheduler = HermesCronSchedulerV3()
    hermes_manager = HermesEpisodicMemoryBackupManagerV3(
        memory_dbs=temp_episodic_dbs_v3, backup_dir=backup_dir_v3, scheduler=scheduler
    )

    hermes_manager.schedule_nightly_episodic_dump("0 3 * * *")

    assert "hermes_nightly_episodic_memory_dump_v3" in scheduler._func_registry

    await hermes_manager.backup_databases()
    files = os.listdir(backup_dir_v3)
    assert len(files) == 2

    await hermes_manager.start()
    await hermes_manager.stop()


@pytest.mark.asyncio
async def test_non_existent_database_handling_v3(
    backup_dir_v3: str, tmp_path: pathlib.Path
) -> None:
    """
    Tests that non-existent databases are safely skipped during backup without raising errors.
    """
    non_existent_path = str(tmp_path / "does_not_exist.db")
    scheduler = HermesCronSchedulerV3()
    manager = SQLiteCronBackupManagerV3(
        databases=[non_existent_path], backup_dir=backup_dir_v3, scheduler=scheduler
    )

    await manager.backup_databases()

    files = os.listdir(backup_dir_v3)
    assert len(files) == 0
    await manager.stop()
