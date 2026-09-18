import sys
from unittest.mock import MagicMock, patch
import pytest

MOCK_MODULES = [
    "chromadb",
    "httpx",
    "yaml",
    "aiohttp",
    "aiofiles",
    "psutil",
    "pydantic",
]

# Ensure missing heavy dependencies are safely patched in sys.modules during test imports
for mod in MOCK_MODULES:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from magda_agent.memory.procedural import ProceduralMemory


@pytest.fixture
def mock_chromadb():
    """Mock chromadb module so tests do not depend on native ChromaDB binary installations."""
    with patch("magda_agent.memory.procedural.chromadb") as mock_chroma:
        mock_ephemeral = MagicMock()
        mock_persistent = MagicMock()
        mock_collection = MagicMock()

        mock_ephemeral.get_or_create_collection.return_value = mock_collection
        mock_persistent.get_or_create_collection.return_value = mock_collection

        mock_chroma.EphemeralClient.return_value = mock_ephemeral
        mock_chroma.PersistentClient.return_value = mock_persistent

        yield {
            "chromadb": mock_chroma,
            "ephemeral_client": mock_ephemeral,
            "persistent_client": mock_persistent,
            "collection": mock_collection,
        }


def test_procedural_memory_init_ephemeral(mock_chromadb):
    pm = ProceduralMemory(persist_directory=":memory:")
    mock_chromadb["chromadb"].EphemeralClient.assert_called_once()
    mock_chromadb["ephemeral_client"].get_or_create_collection.assert_called_once_with(name="procedural_memory")
    assert pm.collection == mock_chromadb["collection"]


def test_procedural_memory_init_persistent(mock_chromadb):
    pm = ProceduralMemory(persist_directory="/tmp/test_procedural_db")
    mock_chromadb["chromadb"].PersistentClient.assert_called_once_with(path="/tmp/test_procedural_db")
    mock_chromadb["persistent_client"].get_or_create_collection.assert_called_once_with(name="procedural_memory")
    assert pm.collection == mock_chromadb["collection"]


def test_store_procedure(mock_chromadb):
    pm = ProceduralMemory(persist_directory=":memory:")
    metadata = {"author": "jules", "category": "automation"}

    pm.store_procedure(
        name="git_commit_push",
        procedure="git add . && git commit -m 'feat' && git push",
        metadata=metadata,
        user_id=42,
    )

    mock_collection = mock_chromadb["collection"]
    mock_collection.add.assert_called_once()
    kwargs = mock_collection.add.call_args.kwargs

    assert len(kwargs["documents"]) == 1
    assert "Procedure Name: git_commit_push" in kwargs["documents"][0]
    assert "git add . && git commit -m 'feat' && git push" in kwargs["documents"][0]

    assert len(kwargs["metadatas"]) == 1
    meta = kwargs["metadatas"][0]
    assert meta["name"] == "git_commit_push"
    assert meta["user_id"] == 42
    assert meta["author"] == "jules"
    assert meta["category"] == "automation"


def test_recall_procedure(mock_chromadb):
    pm = ProceduralMemory(persist_directory=":memory:")
    mock_collection = mock_chromadb["collection"]
    mock_collection.query.return_value = {
        "documents": [
            [
                "Procedure Name: git_commit_push\nProcedure: git add . && git commit",
                "Procedure Name: docker_build\nProcedure: docker build -t app .",
            ]
        ]
    }

    results = pm.recall_procedure(query="How to commit code?", top_k=2, user_id=42)

    mock_collection.query.assert_called_once_with(
        query_texts=["How to commit code?"],
        n_results=2,
        where={"user_id": 42},
    )
    assert len(results) == 2
    assert "git_commit_push" in results[0]
    assert "docker_build" in results[1]


def test_recall_procedure_empty_result(mock_chromadb):
    pm = ProceduralMemory(persist_directory=":memory:")
    mock_collection = mock_chromadb["collection"]
    mock_collection.query.return_value = {"documents": []}

    results = pm.recall_procedure(query="Unknown task")
    assert results == []


def test_get_procedure_versions(mock_chromadb):
    pm = ProceduralMemory(persist_directory=":memory:")
    mock_collection = mock_chromadb["collection"]
    mock_collection.get.return_value = {
        "ids": ["id1", "id2"],
        "documents": ["v1 code", "v2 code"],
        "metadatas": [{"name": "refactor_tool", "v": 1}, {"name": "refactor_tool", "v": 2}],
    }

    # Test without user_id
    versions = pm.get_procedure_versions("refactor_tool")
    mock_collection.get.assert_called_with(where={"name": "refactor_tool"})
    assert len(versions["ids"]) == 2

    # Test with user_id
    versions_user = pm.get_procedure_versions("refactor_tool", user_id=10)
    mock_collection.get.assert_called_with(where={"$and": [{"name": "refactor_tool"}, {"user_id": 10}]})
    assert versions_user == mock_collection.get.return_value


def test_error_handling_graceful(mock_chromadb):
    pm = ProceduralMemory(persist_directory=":memory:")
    mock_collection = mock_chromadb["collection"]
    mock_collection.add.side_effect = RuntimeError("DB connection error")
    mock_collection.query.side_effect = RuntimeError("Query error")
    mock_collection.get.side_effect = RuntimeError("Get error")

    # None of these should crash; they should log and gracefully return fallback values
    pm.store_procedure("proc", "code")
    assert pm.recall_procedure("query") == []
    assert pm.get_procedure_versions("proc") == {}
