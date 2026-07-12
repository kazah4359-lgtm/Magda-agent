import pytest
import os
import shutil
import sqlite3
from pathlib import Path
from magda_agent.scheduler.cron_backups import perform_sqlite_backups

@pytest.mark.asyncio
async def test_perform_sqlite_backups(tmp_path):
    # Change current working directory to tmp_path for the test
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Create a dummy sqlite file
        db_file = tmp_path / "test_data.sqlite3"
        conn = sqlite3.connect(db_file)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        # Create a dummy db file in a subdirectory
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        db_file2 = subdir / "other.db"
        conn2 = sqlite3.connect(db_file2)
        conn2.execute("CREATE TABLE other (id INTEGER PRIMARY KEY)")
        conn2.commit()
        conn2.close()

        # Run backup
        backup_dir = "test_backups"
        await perform_sqlite_backups(backup_dir=backup_dir)

        # Verify backup directory exists
        backup_path = tmp_path / backup_dir
        assert backup_path.exists()
        assert backup_path.is_dir()

        # List files in backup directory
        backup_files = list(backup_path.iterdir())
        assert len(backup_files) == 2

        # Check if names match expected pattern (timestamp is dynamic so we check prefix and suffix)
        file_names = [f.name for f in backup_files]
        assert any(n.startswith("test_data.sqlite3") and n.endswith(".bak") for n in file_names)
        assert any(n.startswith("subdir_other.db") and n.endswith(".bak") for n in file_names)

    finally:
        os.chdir(original_cwd)

@pytest.mark.asyncio
async def test_perform_sqlite_backups_no_files(tmp_path):
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        backup_dir = "test_backups_empty"
        await perform_sqlite_backups(backup_dir=backup_dir)

        # Directory might be created even if no files found (Path.mkdir called before search)
        # but let's check if it's empty if it exists
        backup_path = tmp_path / backup_dir
        if backup_path.exists():
            assert len(list(backup_path.iterdir())) == 0
    finally:
        os.chdir(original_cwd)
