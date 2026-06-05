import chromadb
import uuid
import logging

class SemanticMemory:
    """
    Semantic memory stores stable facts about the project, the world, or the user.
    Uses ChromaDB for vector-based semantic search of facts.
    """
    def __init__(self, persist_directory: str = "./semantic_memory_db") -> None:
        """Initialize SemanticMemory with an ephemeral or persistent ChromaDB client."""
        if persist_directory == ":memory:":
            self.client = chromadb.EphemeralClient()
            logging.info("Initialized SemanticMemory with EphemeralClient")
        else:
            self.client = chromadb.PersistentClient(path=persist_directory)
            logging.info(f"Initialized SemanticMemory with persistent directory: {persist_directory}")
        self.collection = self.client.get_or_create_collection(name="semantic_memory")

    def store_fact(self, text: str, metadata: dict = None, user_id: int = None) -> None:
        """
        Store a stable fact in semantic memory with optional metadata.
        """
        try:
            memory_id = str(uuid.uuid4())

            meta = metadata.copy() if metadata else {}
            if user_id is not None:
                meta["user_id"] = user_id

            if meta:
                self.collection.add(
                    documents=[text],
                    metadatas=[meta],
                    ids=[memory_id]
                )
            else:
                self.collection.add(
                    documents=[text],
                    ids=[memory_id]
                )
            logging.debug(f"Stored semantic fact: {text[:50]}...")
        except Exception as e:
            logging.error(f"Failed to store semantic fact: {e}")

    def recall_facts(self, query: str, top_k: int = 5, user_id: int = None) -> list[str]:
        """
        Recall relevant facts based on semantic similarity to the query.
        """
        try:
            query_kwargs = {
                "query_texts": [query],
                "n_results": top_k
            }
            if user_id is not None:
                query_kwargs["where"] = {"user_id": user_id}

            results = self.collection.query(**query_kwargs)
            if results and results.get("documents") and len(results["documents"]) > 0:
                return results["documents"][0]
            return []
        except Exception as e:
            logging.error(f"Failed to recall semantic facts: {e}")
            return []

    def search_facts(self, query: str, top_k: int = 5, user_id: int = None) -> list[dict]:
        """
        Search facts returning dictionaries with id, text and metadata.
        """
        try:
            query_kwargs = {
                "query_texts": [query],
                "n_results": top_k
            }
            if user_id is not None:
                query_kwargs["where"] = {"user_id": user_id}

            results = self.collection.query(**query_kwargs)
            facts = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                docs = results["documents"][0]
                ids = results["ids"][0]
                metas = results["metadatas"][0] if results.get("metadatas") and results["metadatas"][0] else [{}] * len(docs)
                for i in range(len(docs)):
                    facts.append({
                        "id": ids[i],
                        "text": docs[i],
                        "metadata": metas[i] if metas[i] is not None else {}
                    })
            return facts
        except Exception as e:
            logging.error(f"Failed to search semantic facts: {e}")
            return []

    def delete_fact(self, memory_id: str) -> None:
        """
        Delete a fact from semantic memory by ID.
        """
        try:
            self.collection.delete(ids=[memory_id])
            logging.debug(f"Deleted semantic fact with ID: {memory_id}")
        except Exception as e:
            logging.error(f"Failed to delete semantic fact: {e}")
