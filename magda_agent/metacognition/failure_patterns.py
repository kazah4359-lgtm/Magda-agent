import json
import logging
import sqlite3
import datetime
from typing import Optional, Dict, Any, List

from magda_agent.llm_client import LLMClient

class FailurePatternTracker:
    """
    Tracks failures (test failures, user corrections, plan abandonment).
    Detects recurring patterns and generates preventive rules.
    """
    def __init__(self, llm: LLMClient, db_path: str = ":memory:"):
        """
        Initializes the failure tracker.

        Args:
            llm: LLMClient instance to generate preventive rules.
            db_path: Path to SQLite DB. Uses in-memory by default.
        """
        self.llm = llm
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initializes the sqlite schema."""
        cursor = self._conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                category TEXT,
                detail TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preventive_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                rule TEXT,
                timestamp TEXT
            )
        ''')
        self._conn.commit()

    def log_failure(self, category: str, detail: str) -> None:
        """
        Logs a failure event.

        Args:
            category: The category of the failure.
            detail: Detailed description of the failure.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            'INSERT INTO failures (timestamp, category, detail) VALUES (?, ?, ?)',
            (datetime.datetime.now().isoformat(), category, detail)
        )
        self._conn.commit()

    def _get_failures_by_category(self, category: str) -> List[str]:
        """Gets failure details for a category."""
        cursor = self._conn.cursor()
        cursor.execute('SELECT detail FROM failures WHERE category = ? ORDER BY timestamp DESC LIMIT 10', (category,))
        rows = cursor.fetchall()
        return [row[0] for row in rows]

    async def detect_recurring_patterns(self, threshold: int = 2) -> List[Dict[str, Any]]:
        """
        Detects if a failure category repeats more than `threshold` times
        and generates preventive rules.

        Args:
            threshold: Number of occurrences to trigger pattern detection.

        Returns:
            List of generated preventive rules with their category.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            'SELECT category, COUNT(*) as count FROM failures GROUP BY category HAVING count > ?',
            (threshold,)
        )
        recurring = cursor.fetchall()

        results = []
        for row in recurring:
            category = row[0]
            # Check if we already generated a rule recently to prevent spamming
            cursor.execute('SELECT id FROM preventive_rules WHERE category = ?', (category,))
            if cursor.fetchone():
                continue

            failures = self._get_failures_by_category(category)
            rule = await self._generate_preventive_rule(category, failures)
            if rule:
                cursor.execute(
                    'INSERT INTO preventive_rules (category, rule, timestamp) VALUES (?, ?, ?)',
                    (category, rule, datetime.datetime.now().isoformat())
                )
                self._conn.commit()
                results.append({"category": category, "rule": rule})

        return results

    async def _generate_preventive_rule(self, category: str, failures: List[str]) -> Optional[str]:
        """
        Calls the LLM to generate a preventive rule for the recurring failures.
        """
        failure_text = "\n".join(f"- {f}" for f in failures)
        prompt = (
            f"The following failures occurred repeatedly in the '{category}' category:\n"
            f"{failure_text}\n\n"
            "Analyze these failures and provide a single concise, actionable preventive rule "
            "to avoid this in the future."
        )
        messages = [{"role": "system", "content": prompt}]

        try:
            rule_text = await self.llm.chat_completion(messages, temperature=0.1)
            return rule_text.strip()
        except Exception as e:
            logging.error(f"Failed to generate preventive rule for category '{category}': {e}")
            return None

    def get_preventive_rules(self) -> List[str]:
        """
        Retrieves all active preventive rules.
        """
        cursor = self._conn.cursor()
        cursor.execute('SELECT rule FROM preventive_rules ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        return [row[0] for row in rows]
