import logging
from magda_agent.memory.episodic import EpisodicMemory

class LongTermMemory:
    """
    Legacy wrapper for long-term memory operations, now delegating to EpisodicMemory.
    Stores conversations and other textual experiences.
    Maintained for backwards compatibility with existing API.
    """
    def __init__(self, persist_directory: str = "./memory_db"):
        self.episodic_memory = EpisodicMemory(persist_directory=persist_directory)
        logging.info("Initialized LongTermMemory acting as wrapper for EpisodicMemory")

    def store(self, text: str, metadata: dict = None, user_id: int = None) -> None:
        """
        Store a textual memory with optional metadata, delegating to episodic memory.
        """
        self.episodic_memory.store_event(text, metadata=metadata, user_id=user_id)

    def recall(self, query: str, top_k: int = 5, user_id: int = None) -> list[str]:
        """
        Recall relevant memories based on semantic similarity, delegating to episodic memory.
        """
        return self.episodic_memory.recall_events(query, top_k=top_k, user_id=user_id)
