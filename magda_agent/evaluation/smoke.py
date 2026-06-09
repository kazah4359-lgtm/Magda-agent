import subprocess
import logging
from typing import List, Dict, Any, Optional

class SmokeEvaluator:
    """
    Evaluates codebase dynamically via smoke testing after new code is merged.
    """
    def __init__(self, command: str = "python3 -m pytest tests/", cwd: Optional[str] = None):
        self.command = command
        self.cwd = cwd

    def evaluate(self) -> Dict[str, Any]:
        """
        Runs the configured smoke test command and returns the evaluation result.
        """
        logging.info(f"Running smoke evaluation command: {self.command}")

        try:
            result = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.cwd
            )

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": self.command
            }
        except Exception as e:
            logging.error(f"Smoke evaluation failed: {str(e)}")
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "command": self.command
            }
