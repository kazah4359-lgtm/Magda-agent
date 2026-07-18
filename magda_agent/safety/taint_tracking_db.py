"""Persistent storage for tainted tool execution logs."""
import sqlite3
import json
import time
from typing import Any, Callable, Dict, List, Optional
from magda_agent.safety.taint import MCPKernel, PolicyViolationError

class TaintTrackingDB:
    """Manages an SQLite database for storing taint tracking logs."""

    def __init__(self, db_path: str = "taint_tracking.db", timeout: float = 10.0):
        self.db_path = db_path
        self.timeout = timeout
        self._init_db()

    def _init_db(self) -> None:
        """Initializes the database schema."""
        if self.db_path == ":memory:":
            return # Let the caller manage memory DB connections

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS taint_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    tool_name TEXT NOT NULL,
                    inputs TEXT NOT NULL,
                    is_sensitive BOOLEAN NOT NULL,
                    violation_raised BOOLEAN NOT NULL,
                    error_message TEXT
                )
                '''
            )
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection."""
        # Using timeout for concurrent access
        return sqlite3.connect(self.db_path, timeout=self.timeout)

    def log_execution(self, tool_name: str, inputs: Dict[str, Any], is_sensitive: bool, violation_raised: bool, error_message: Optional[str] = None, conn: Optional[sqlite3.Connection] = None) -> None:
        """Logs an execution attempt."""
        timestamp = time.time()

        # Simple serialization; in real systems use recursive sanitization,
        # but here we follow the task requirements. We stringify safely.
        try:
            # We convert it to a string rather than full json dump to handle TaintedString objects
            inputs_str = str(inputs)
        except Exception:
            inputs_str = "<unserializable>"

        insert_query = '''
            INSERT INTO taint_logs (timestamp, tool_name, inputs, is_sensitive, violation_raised, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
        '''
        params = (timestamp, tool_name, inputs_str, is_sensitive, violation_raised, error_message)

        if conn is not None:
             cursor = conn.cursor()
             # If caller passed a connection, use it (useful for :memory: testing)
             cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS taint_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    tool_name TEXT NOT NULL,
                    inputs TEXT NOT NULL,
                    is_sensitive BOOLEAN NOT NULL,
                    violation_raised BOOLEAN NOT NULL,
                    error_message TEXT
                )
                '''
             )
             cursor.execute(insert_query, params)
             conn.commit()
             return

        with self._get_connection() as c:
            cursor = c.cursor()
            cursor.execute(insert_query, params)
            c.commit()

    def query(self, tool_name: Optional[str] = None, conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
        """Queries the logs."""
        query_sql = "SELECT id, timestamp, tool_name, inputs, is_sensitive, violation_raised, error_message FROM taint_logs"
        params: List[Any] = []

        if tool_name:
            query_sql += " WHERE tool_name = ?"
            params.append(tool_name)

        def execute_query(c: sqlite3.Connection):
            cursor = c.cursor()
            # ensure table exists for memory db if query is called first
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS taint_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    tool_name TEXT NOT NULL,
                    inputs TEXT NOT NULL,
                    is_sensitive BOOLEAN NOT NULL,
                    violation_raised BOOLEAN NOT NULL,
                    error_message TEXT
                )
                '''
            )
            cursor.execute(query_sql, tuple(params))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "tool_name": row[2],
                    "inputs": row[3],
                    "is_sensitive": bool(row[4]),
                    "violation_raised": bool(row[5]),
                    "error_message": row[6]
                })
            return results

        if conn is not None:
            return execute_query(conn)

        with self._get_connection() as c:
            return execute_query(c)

def execute_with_tracking(kernel: MCPKernel, db: TaintTrackingDB, tool_name: str, tool_func: Callable[..., Any], inputs: Dict[str, Any], is_sensitive: bool = False, conn: Optional[sqlite3.Connection] = None) -> Any:
    """Wraps MCPKernel execute_tool with DB tracking."""
    violation_raised = False
    error_message = None

    try:
        result = kernel.execute_tool(tool_func, inputs, is_sensitive=is_sensitive)
        return result
    except PolicyViolationError as e:
        violation_raised = True
        error_message = str(e)
        raise
    except Exception as e:
        error_message = str(e)
        raise
    finally:
        # We always log the attempt, regardless of whether it succeeded, failed policy, or failed otherwise.
        # But we only mark violation_raised if it was a PolicyViolationError
        db.log_execution(
            tool_name=tool_name,
            inputs=inputs,
            is_sensitive=is_sensitive,
            violation_raised=violation_raised,
            error_message=error_message,
            conn=conn
        )
