import os
import shutil
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

async def perform_sqlite_backups(backup_dir: str = "backups"):
    """
    Scans the project for SQLite databases (.sqlite3, .db) and
    copies them to a backup directory with a timestamp.
    Inspired by Hermes Agent trend for persistent state protection.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)

        # Files to search for (common extensions)
        extensions = [".sqlite3", ".db"]

        found_databases = []
        for root, dirs, files in os.walk("."):
            # Skip the backup directory and common hidden/temp dirs to avoid recursion or backing up junk
            if backup_dir in root or ".git" in root or "__pycache__" in root or ".pytest_cache" in root or "venv" in root:
                continue

            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    found_databases.append(Path(root) / file)

        if not found_databases:
            logger.info("No SQLite databases found for backup.")
            return

        for db_file in found_databases:
            # Create a safe name for the backup: flatten path and add timestamp
            # e.g. ./subdir/data.db -> subdir_data.db.20231027_020000.bak
            try:
                relative_path = db_file.relative_to(".")
                safe_name = str(relative_path).replace(os.sep, "_")
                backup_filename = f"{safe_name}.{timestamp}.bak"
                target_path = backup_path / backup_filename

                logger.info(f"Backing up {db_file} to {target_path}")
                shutil.copy2(db_file, target_path)
            except Exception as fe:
                logger.error(f"Failed to backup individual file {db_file}: {fe}")

        logger.info(f"Successfully backed up {len(found_databases)} databases to {backup_dir}")

    except Exception as e:
        logger.error(f"Failed to perform SQLite backups: {e}", exc_info=True)
