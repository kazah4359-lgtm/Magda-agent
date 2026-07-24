import sqlite3
import threading
import time
import os
import pytest
from magda_agent.utils.sqlite_pool import get_connection

def test_sqlite_pool_concurrent_writes(tmp_path):
    db_path = str(tmp_path / "test.db")

    # Initialize DB
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
        conn.commit()

    def write_db(worker_id):
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            for i in range(10):
                cursor.execute("INSERT INTO test (val) VALUES (?)", (f"worker_{worker_id}_{i}",))
                conn.commit()
                time.sleep(0.01)

    threads = []
    for i in range(5):
        t = threading.Thread(target=write_db, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test")
        count = cursor.fetchone()[0]
        assert count == 50
