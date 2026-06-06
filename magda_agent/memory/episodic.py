import math
import chromadb
import uuid
import logging

class EpisodicMemory:
    """
    Episodic memory stores events (like conversations) chronologically.
    Uses ChromaDB for vector-based semantic search of past episodes.
    """
    def __init__(self, persist_directory: str = "./episodic_memory_db") -> None:
        """Initialize EpisodicMemory with an ephemeral or persistent ChromaDB client."""
        if persist_directory == ":memory:":
            self.client = chromadb.EphemeralClient()
            logging.info("Initialized EpisodicMemory with EphemeralClient")
        else:
            self.client = chromadb.PersistentClient(path=persist_directory)
            logging.info(f"Initialized EpisodicMemory with persistent directory: {persist_directory}")
        self.collection = self.client.get_or_create_collection(name="episodic_memory")

    def store_event(self, text: str, metadata: dict = None, user_id: int = None) -> None:
        """
        Store an event in episodic memory with optional metadata.
        """
        try:
            memory_id = str(uuid.uuid4())

            meta = metadata.copy() if metadata else {}
            if user_id is not None:
                meta["user_id"] = user_id

            # Ensure decayed field is present for filtering later
            if "decayed" not in meta:
                meta["decayed"] = False

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
            logging.debug(f"Stored episodic event: {text[:50]}...")
        except Exception as e:
            logging.error(f"Failed to store episodic event: {e}")

    def decay_event(self, memory_id: str) -> None:
        """Mark an event as decayed."""
        try:
            res = self.collection.get(ids=[memory_id])
            if res and res["metadatas"] and len(res["metadatas"]) > 0:
                meta = res["metadatas"][0]
                meta["decayed"] = True
                self.collection.update(ids=[memory_id], metadatas=[meta])
                logging.debug(f"Decayed episodic event: {memory_id}")
        except Exception as e:
            logging.error(f"Failed to decay episodic event: {e}")

    def get_all_events(self, user_id: int = None, include_decayed: bool = False, limit: int = 100) -> list[dict]:
        """Get all events, optionally filtering by user_id and decay status."""
        try:
            where = {}
            if user_id is not None:
                where["user_id"] = user_id
            if not include_decayed:
                where["decayed"] = False

            if where:
                if len(where) > 1:
                    where_list = [{k: v} for k, v in where.items()]
                    results = self.collection.get(where={"$and": where_list}, limit=limit)
                else:
                    results = self.collection.get(where=where, limit=limit)
            else:
                results = self.collection.get(limit=limit)

            events = []
            if results and results.get("documents"):
                for i in range(len(results["documents"])):
                    events.append({
                        "id": results["ids"][i],
                        "text": results["documents"][i],
                        "metadata": results["metadatas"][i] if results.get("metadatas") else {}
                    })
            return events
        except Exception as e:
            logging.error(f"Failed to get episodic events: {e}")
            return []

    def recall_events(self, query: str, top_k: int = 5, user_id: int = None) -> list[str]:
        """
        Recall relevant events based on semantic similarity to the query.
        """
        try:
            query_kwargs = {
                "query_texts": [query],
                "n_results": top_k * 2
            }
            where_clause = {"decayed": False}
            if user_id is not None:
                where_clause["user_id"] = user_id

            # If multiple conditions, ChromaDB needs $and. However, since we default to false, we can just query it.
            if len(where_clause) > 1:
                query_kwargs["where"] = {"$and": [{"user_id": user_id}, {"decayed": False}]}
            else:
                query_kwargs["where"] = where_clause

            results = self.collection.query(**query_kwargs)
            if results and results.get("documents") and len(results["documents"]) > 0:
                docs = results["documents"][0]
                dists = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)
                metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(docs)

                scored_docs = []
                for doc, dist, meta in zip(docs, dists, metas):
                    meta = meta or {}
                    pad_p = float(meta.get("pad_p", 0.0))
                    pad_a = float(meta.get("pad_a", 0.0))
                    pad_d = float(meta.get("pad_d", 0.0))

                    intensity = math.sqrt(pad_p**2 + pad_a**2 + pad_d**2)
                    adjusted_score = dist - (intensity * 1.0)
                    scored_docs.append((adjusted_score, doc))

                scored_docs.sort(key=lambda x: x[0])

                return [doc for score, doc in scored_docs[:top_k]]
            return []
        except Exception as e:
            logging.error(f"Failed to recall episodic events: {e}")
            return []
