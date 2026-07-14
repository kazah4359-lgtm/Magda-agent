"""MCPKernel Taint Tracking Sandbox V3.

Provides an enhanced taint tracking system (TaintTrackerV3) and an integrated
memory block (TaintedEpisodicMemory) to preserve and trace data provenance
across memory boundaries (episodic memory).
"""
import logging
import math
from typing import Any, Dict, List, Optional, Set

from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.safety.taint_tracking_v2 import TaintTrackerV2


class TaintTrackerV3(TaintTrackerV2):
    """Enhanced TaintTracker that adds support for tracing data provenance across episodic and other memory blocks explicitly."""

    def __init__(self) -> None:
        """Initialize the TaintTrackerV3."""
        super().__init__()


class TaintedEpisodicMemory(EpisodicMemory):
    """An enhanced EpisodicMemory that integrates with TaintTrackerV3 to maintain taint tracks

    and trace data provenance when data is stored and retrieved.
    """

    def __init__(self, persist_directory: str = "./episodic_memory_db", tracker: Optional[TaintTrackerV3] = None) -> None:
        """Initialize the TaintedEpisodicMemory.

        Args:
            persist_directory: The directory path for ChromaDB storage.
            tracker: An optional TaintTrackerV3 instance.
        """
        super().__init__(persist_directory=persist_directory)
        self.tracker = tracker or TaintTrackerV3()

    def store_event(self, text: str, metadata: Optional[dict] = None, user_id: Optional[int] = None) -> None:
        """Store an event in episodic memory, preserving its taint tracks in the metadata.

        Args:
            text: The text event content, potentially containing tainted data.
            metadata: Optional dictionary of metadata associated with the event.
            user_id: Optional user identifier.
        """
        if metadata is None:
            metadata = {}
        else:
            metadata = metadata.copy()

        # If text is tainted, capture its origins
        if self.tracker.is_tainted(text):
            origins = list(self.tracker.get_origins(text))
            # ChromaDB supports string, int, float, bool. It does NOT support lists in metadata values.
            # So we serialize the origins list into a comma-separated string.
            metadata["_taint_origins"] = ",".join(origins)
            metadata["_is_tainted"] = True

        # Sanitize everything to standard python types before storing
        clean_text = self.tracker.sanitize(text)
        clean_metadata = self.tracker.sanitize(metadata)

        super().store_event(clean_text, clean_metadata, user_id)

    def get_all_events(self, user_id: Optional[int] = None, include_decayed: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve all events, reconstructing any tainted text from stored origins.

        Args:
            user_id: Optional user identifier to filter by.
            include_decayed: Whether to include decayed events.
            limit: Maximum number of events to retrieve.

        Returns:
            A list of event dictionaries with reconstructed tainted text.
        """
        events = super().get_all_events(user_id=user_id, include_decayed=include_decayed, limit=limit)
        for event in events:
            meta = event.get("metadata", {})
            if meta and meta.get("_is_tainted") and "_taint_origins" in meta:
                origins_str = meta["_taint_origins"]
                origins = set(origins_str.split(",")) if origins_str else set()
                event["text"] = self.tracker.taint_with_origins(event["text"], origins)
        return events

    def recall_events(self, query: str, top_k: int = 5, user_id: Optional[int] = None) -> List[str]:
        """Recall relevant events, reconstructing taint tracks on returned strings.

        Args:
            query: Semantic search query string.
            top_k: Number of retrieved results to return.
            user_id: Optional user identifier to filter by.

        Returns:
            A list of recalled strings, with reconstructed taint tracking where applicable.
        """
        try:
            query_kwargs = {
                "query_texts": [query],
                "n_results": top_k * 2
            }
            where_clause = {"decayed": False}
            if user_id is not None:
                where_clause["user_id"] = user_id

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
                    scored_docs.append((adjusted_score, doc, meta))

                scored_docs.sort(key=lambda x: x[0])

                recalled = []
                for _, doc, meta in scored_docs[:top_k]:
                    if meta and meta.get("_is_tainted") and "_taint_origins" in meta:
                        origins_str = meta["_taint_origins"]
                        origins = set(origins_str.split(",")) if origins_str else set()
                        doc = self.tracker.taint_with_origins(doc, origins)
                    recalled.append(doc)
                return recalled
            return []
        except Exception as e:
            logging.error(f"Failed to recall episodic events in TaintedEpisodicMemory: {e}")
            return []
