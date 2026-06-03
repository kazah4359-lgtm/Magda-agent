import chromadb
from chromadb.config import Settings
import uuid
import logging

class LongTermMemory:
    """
    Hippocampus: Long-term memory module using ChromaDB for semantic search.
    Stores conversations, facts, and other textual experiences.
    """
    def __init__(self, persist_directory: str = "./memory_db"):
        if persist_directory == ":memory:":
            self.client = chromadb.EphemeralClient()
            logging.info("Initialized LongTermMemory with EphemeralClient")
        else:
            self.client = chromadb.PersistentClient(path=persist_directory)
            logging.info(f"Initialized LongTermMemory with persistent directory: {persist_directory}")
        self.collection = self.client.get_or_create_collection(name="long_term_memory")

    def store(self, text: str, metadata: dict = None) -> None:
        """
        Store a textual memory with optional metadata.
        """
        try:
            memory_id = str(uuid.uuid4())
            if metadata:
                self.collection.add(
                    documents=[text],
                    metadatas=[metadata],
                    ids=[memory_id]
                )
            else:
                self.collection.add(
                    documents=[text],
                    ids=[memory_id]
                )
            logging.debug(f"Stored memory: {text[:50]}...")
        except Exception as e:
            logging.error(f"Failed to store memory: {e}")

    def recall(self, query: str, top_k: int = 5) -> list[str]:
        """
        Recall relevant memories based on the semantic similarity to the query.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            if results and results.get("documents") and len(results["documents"]) > 0:
                return results["documents"][0]
            return []
        except Exception as e:
            logging.error(f"Failed to recall memories: {e}")
            return []
