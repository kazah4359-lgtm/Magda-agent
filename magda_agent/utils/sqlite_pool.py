import sqlite3
import threading
import time
from typing import Optional

class SQLiteConnectionManager:
    """
    A simple connection manager with retry logic to handle SQLite 'database is locked'
    errors during concurrent access. It wraps the sqlite3.connect context manager.
    """
    def __init__(self, db_path: str, timeout: float = 10.0, max_retries: int = 5, retry_delay: float = 0.5):
        self.db_path = db_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def __enter__(self):
        retries = 0
        while retries < self.max_retries:
            try:
                # Use isolation_level=None for autocommit, or keep default
                self.conn = sqlite3.connect(self.db_path, timeout=self.timeout)
                return self.conn
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    retries += 1
                    time.sleep(self.retry_delay)
                else:
                    raise e
        raise sqlite3.OperationalError(f"Database at {self.db_path} is locked after {self.max_retries} retries.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

def get_connection(db_path: str, timeout: float = 10.0) -> SQLiteConnectionManager:
    return SQLiteConnectionManager(db_path, timeout=timeout)
