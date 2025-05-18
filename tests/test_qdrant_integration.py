import uuid

import pytest
from mcp_server_qdrant.embeddings.ollama_provider import OllamaEmbedProvider
from mcp_server_qdrant.qdrant import Entry, QdrantConnector


@pytest.fixture
async def embedding_provider():
    return  OllamaEmbedProvider(model_name="bge-m3")


@pytest.fixture
async def qdrant_connector(embedding_provider):
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
    )
    yield connector


@pytest.mark.asyncio
async def test_store_and_search(qdrant_connector):
    test_entry = Entry(
        content="The quick brown fox jumps over the lazy dog",
        payload={"document": "The quick brown fox jumps over the lazy dog", "metadata": {"source": "test", "importance": "high"}},
        metadata={"source": "test", "importance": "high"},
    )
    await qdrant_connector.store(test_entry)

    results = await qdrant_connector.search("fox jumps")
    assert len(results) == 1
    assert results[0].payload["document"] == test_entry.content
    assert results[0].payload["metadata"] == test_entry.metadata


@pytest.mark.asyncio
async def test_search_empty_collection(qdrant_connector):
    results = await qdrant_connector.search("test query")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_multiple_entries(qdrant_connector):
    entries = [
        Entry(
            content="Python is a programming language",
            payload={"document": "Python is a programming language", "metadata": {"topic": "programming"}},
            metadata={"topic": "programming"},
        ),
        Entry(
            content="The Eiffel Tower is in Paris",
            payload={"document": "The Eiffel Tower is in Paris", "metadata": {"topic": "landmarks"}},
            metadata={"topic": "landmarks"},
        ),
        Entry(
            content="Machine learning is a subset of AI",
            payload={"document": "Machine learning is a subset of AI", "metadata": {"topic": "AI"}},
            metadata={"topic": "AI"},
        ),
    ]

    for entry in entries:
        await qdrant_connector.store(entry)

    programming_results = await qdrant_connector.search("Python programming")
    assert len(programming_results) > 0
    assert any("Python" in result.payload["document"] for result in programming_results)

    landmark_results = await qdrant_connector.search("Eiffel Tower Paris")
    assert len(landmark_results) > 0
    assert any("Eiffel" in result.payload["document"] for result in landmark_results)

    ai_results = await qdrant_connector.search("artificial intelligence machine learning")
    assert len(ai_results) > 0
    assert any("machine learning" in result.payload["document"].lower() for result in ai_results)


@pytest.mark.asyncio
async def test_ensure_collection_exists(qdrant_connector):
    assert not await qdrant_connector._client.collection_exists(qdrant_connector._default_collection_name)
    test_entry = Entry(content="Test content", payload={"document": "Test content"})
    await qdrant_connector.store(test_entry)
    assert await qdrant_connector._client.collection_exists(qdrant_connector._default_collection_name)


@pytest.mark.asyncio
async def test_metadata_handling(qdrant_connector):
    metadata1 = {"source": "book", "author": "Jane Doe", "year": 2023}
    metadata2 = {"source": "article", "tags": ["science", "research"]}

    await qdrant_connector.store(Entry(content="Content with structured metadata", payload={"document": "Content with structured metadata", "metadata": metadata1}, metadata=metadata1))
    await qdrant_connector.store(Entry(content="Content with list in metadata", payload={"document": "Content with list in metadata", "metadata": metadata2}, metadata=metadata2))

    results = await qdrant_connector.search("metadata")
    assert len(results) == 2

    found_metadata1 = False
    found_metadata2 = False

    for result in results:
        meta = result.payload.get("metadata", {})
        if meta.get("source") == "book":
            assert meta.get("author") == "Jane Doe"
            assert meta.get("year") == 2023
            found_metadata1 = True
        elif meta.get("source") == "article":
            assert "science" in meta.get("tags", [])
            assert "research" in meta.get("tags", [])
            found_metadata2 = True

    assert found_metadata1
    assert found_metadata2


@pytest.mark.asyncio
async def test_entry_without_metadata(qdrant_connector):
    await qdrant_connector.store(Entry(content="Entry without metadata", payload={"document": "Entry without metadata"}))
    results = await qdrant_connector.search("without metadata")
    assert len(results) == 1
    assert results[0].payload["document"] == "Entry without metadata"


@pytest.mark.asyncio
async def test_custom_collection_store_and_search(qdrant_connector):
    custom_collection = f"custom_collection_{uuid.uuid4().hex}"

    test_entry = Entry(
        content="This is stored in a custom collection",
        payload={"document": "This is stored in a custom collection", "metadata": {"custom": True}},
        metadata={"custom": True},
    )
    await qdrant_connector.store(test_entry, collection_name=custom_collection)

    results = await qdrant_connector.search("custom collection", collection_name=custom_collection)
    assert len(results) == 1
    assert results[0].payload["document"] == test_entry.content
    assert results[0].payload["metadata"] == test_entry.metadata

    default_results = await qdrant_connector.search("custom collection")
    assert len(default_results) == 0


@pytest.mark.asyncio
async def test_multiple_collections(qdrant_connector):
    collection_a = f"collection_a_{uuid.uuid4().hex}"
    collection_b = f"collection_b_{uuid.uuid4().hex}"

    entry_a = Entry(content="This belongs to collection A", payload={"document": "This belongs to collection A", "collection": "A"})
    entry_b = Entry(content="This belongs to collection B", payload={"document": "This belongs to collection B", "collection": "B"})
    entry_default = Entry(content="This belongs to the default collection", payload={"document": "This belongs to the default collection"})

    await qdrant_connector.store(entry_a, collection_name=collection_a)
    await qdrant_connector.store(entry_b, collection_name=collection_b)
    await qdrant_connector.store(entry_default)

    results_a = await qdrant_connector.search("belongs", collection_name=collection_a)
    assert len(results_a) == 1
    assert results_a[0].payload["document"] == entry_a.content

    results_b = await qdrant_connector.search("belongs", collection_name=collection_b)
    assert len(results_b) == 1
    assert results_b[0].payload["document"] == entry_b.content

    results_default = await qdrant_connector.search("belongs")
    assert len(results_default) == 1
    assert results_default[0].payload["document"] == entry_default.content


@pytest.mark.asyncio
async def test_nonexistent_collection_search(qdrant_connector):
    nonexistent_collection = f"nonexistent_{uuid.uuid4().hex}"
    results = await qdrant_connector.search("test query", collection_name=nonexistent_collection)
    assert len(results) == 0