import time
from typing import List, Dict, Any

class ThoughtChainTracer:
    """
    Traces the internal reasoning steps of the cognitive architecture.
    """
    def __init__(self):
        self.trace: List[Dict[str, Any]] = []

    def add_step(self, step_name: str, data: Any = None) -> None:
        """
        Records a single cognitive step.
        """
        entry = {
            "timestamp": time.time(),
            "step": step_name
        }
        if data is not None:
            entry["data"] = data

        self.trace.append(entry)

    def get_trace(self) -> List[Dict[str, Any]]:
        """
        Returns the full current trace.
        """
        return self.trace

    def clear(self) -> None:
        """
        Clears the current trace.
        """
        self.trace.clear()
