from magda_agent.memory.persistent_v3 import PersistentMemoryLayerV3

class MockDBClient:
    def __init__(self):
        self.saved_chunks = []

    def save(self, chunk):
        self.saved_chunks.append(chunk)

def test_persistent_memory_layer_v3_on_context_update() -> None:
    mock_db = MockDBClient()
    plugin = PersistentMemoryLayerV3(db_client=mock_db)

    user_id = 123
    new_context = "User said hello"

    plugin.on_context_update(new_context, user_id)

    assert len(mock_db.saved_chunks) == 1
    assert mock_db.saved_chunks[0]["user_id"] == 123
    assert mock_db.saved_chunks[0]["context"] == "User said hello"
    assert len(plugin.persisted_chunks) == 1
