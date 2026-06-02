import time
import json
from typing import List, Dict, Optional

class MemoryEntry:
    def __init__(self, content: str, importance: float, emotion: Dict[str, float], tags: List[str]):
        self.content = content
        self.importance = importance  # 0.0 to 1.0
        self.emotion = emotion        # e.g., {"pleasure": 0.5, "arousal": 0.2}
        self.tags = tags
        self.timestamp = time.time()
        self.recall_count = 0

class LongTermMemory:
    def __init__(self, storage_path: str = "memory.json"):
        self.storage_path = storage_path
        self.memories: List[MemoryEntry] = []
        self.load()

    def add_memory(self, content: str, importance: float, emotion: Dict[str, float], tags: List[str]):
        entry = MemoryEntry(content, importance, emotion, tags)
        self.memories.append(entry)
        self.save()

    def search_memories(self, query_tags: List[str]) -> List[MemoryEntry]:
        # Simple tag-based search for now
        results = [m for m in self.memories if any(tag in m.tags for tag in query_tags)]
        for res in results:
            res.recall_count += 1
        return sorted(results, key=lambda x: x.importance, reverse=True)

    def consolidate(self):
        """
        Subconsciousness calls this to decay old/unimportant memories
        and reinforce important ones.
        """
        current_time = time.time()
        new_memories = []
        for m in self.memories:
            # Simple decay logic: importance decreases over time unless recalled often
            age = (current_time - m.timestamp) / 3600  # age in hours
            decay_factor = 0.95 ** age
            effective_importance = m.importance * decay_factor + (m.recall_count * 0.1)

            if effective_importance > 0.1:
                m.importance = min(1.0, effective_importance)
                new_memories.append(m)

        self.memories = new_memories
        self.save()

    def save(self):
        data = [
            {
                "content": m.content,
                "importance": m.importance,
                "emotion": m.emotion,
                "tags": m.tags,
                "timestamp": m.timestamp,
                "recall_count": m.recall_count
            }
            for m in self.memories
        ]
        with open(self.storage_path, "w") as f:
            json.dump(data, f)

    def load(self):
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                self.memories = []
                for item in data:
                    m = MemoryEntry(item["content"], item["importance"], item["emotion"], item["tags"])
                    m.timestamp = item["timestamp"]
                    m.recall_count = item["recall_count"]
                    self.memories.append(m)
        except (FileNotFoundError, json.JSONDecodeError):
            self.memories = []
