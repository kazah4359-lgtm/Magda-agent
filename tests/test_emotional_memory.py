import pytest
from magda_agent.memory.episodic import EpisodicMemory

def test_emotional_memory_tagging_and_boosting():
    # Use ephemeral client for testing
    memory = EpisodicMemory(persist_directory=":memory:")

    # Generate unique collection for this test to avoid state sharing
    import uuid
    col_name = f"episodic_{uuid.uuid4().hex}"
    memory.collection = memory.client.create_collection(col_name)

    # Add a neutral memory
    memory.store_event(
        text="I walked in the park.",
        metadata={"user_id": 1, "pad_p": 0.0, "pad_a": 0.0, "pad_d": 0.0},
        user_id=1
    )

    # Add an emotionally charged memory with very similar text
    memory.store_event(
        text="I walked in the park and was terrified by a bear.",
        metadata={"user_id": 1, "pad_p": -0.8, "pad_a": 0.9, "pad_d": -0.5},
        user_id=1
    )

    # Search for "walked in the park"
    results = memory.recall_events("I walked in the park", top_k=1, user_id=1)

    # The emotionally charged memory should be boosted and come out on top
    # despite the neutral memory being a closer match to "walked in the park" text-wise.
    assert len(results) == 1
    assert "bear" in results[0]

def test_store_event_with_metadata():
    memory = EpisodicMemory(persist_directory=":memory:")
    import uuid
    col_name = f"episodic_{uuid.uuid4().hex}"
    memory.collection = memory.client.create_collection(col_name)

    memory.store_event(
        text="A test event",
        metadata={"pad_p": 1.0, "pad_a": 1.0, "pad_d": 1.0},
        user_id=2
    )

    all_events = memory.get_all_events(user_id=2)
    assert len(all_events) == 1
    assert all_events[0]["metadata"]["pad_p"] == 1.0
