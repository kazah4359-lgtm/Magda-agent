import chromadb
import uuid
import logging

class ProceduralMemory:
    """
    Procedural memory stores reusable successful methods and procedures.
    Uses ChromaDB for vector-based semantic retrieval of procedures.
    """
    def __init__(self, persist_directory: str = "./procedural_memory_db") -> None:
        """Initialize ProceduralMemory with an ephemeral or persistent ChromaDB client."""
        if persist_directory == ":memory:":
            self.client = chromadb.EphemeralClient()
            logging.info("Initialized ProceduralMemory with EphemeralClient")
        else:
            self.client = chromadb.PersistentClient(path=persist_directory)
            logging.info(f"Initialized ProceduralMemory with persistent directory: {persist_directory}")
        self.collection = self.client.get_or_create_collection(name="procedural_memory")

    def store_procedure(self, name: str, procedure: str, metadata: dict = None, user_id: int = None) -> None:
        """
        Store a procedural memory (e.g., a method or steps) with optional metadata.
        """
        try:
            memory_id = str(uuid.uuid4())

            meta = metadata.copy() if metadata else {}
            meta["name"] = name
            if user_id is not None:
                meta["user_id"] = user_id

            # The content being embedded is the name and the procedure for better semantic matching
            content = f"Procedure Name: {name}\nProcedure: {procedure}"

            self.collection.add(
                documents=[content],
                metadatas=[meta],
                ids=[memory_id]
            )
            logging.debug(f"Stored procedure: {name}")
        except Exception as e:
            logging.error(f"Failed to store procedure: {e}")

    def recall_procedure(self, query: str, top_k: int = 5, user_id: int = None) -> list[str]:
        """
        Recall relevant procedures based on semantic similarity to the query.
        Returns the procedural documents that match.
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
            logging.error(f"Failed to recall procedures: {e}")
            return []

    def get_procedure_versions(self, name: str, user_id: int = None) -> dict:
        """
        Retrieve all versions of a skill procedure by name.
        """
        try:
            where_clause = {"name": name}
            if user_id is not None:
                where_clause = {"$and": [{"name": name}, {"user_id": user_id}]}

            results = self.collection.get(where=where_clause)
            return results
        except Exception as e:
            logging.error(f"Failed to get procedure versions: {e}")
            return {}
