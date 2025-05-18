# tests/test_store_qdrant.py

import pytest
import asyncio
import uuid

from mcp_server_qdrant.qdrant import QdrantConnector, Entry

# Mock del embedding provider
class MockEmbeddingProvider:
    def get_vector_name(self):
        return "default"

    def get_vector_size(self):
        return 1536

    async def embed_documents(self, texts):
        return [[0.1] * self.get_vector_size() for _ in texts]

    async def embed_query(self, query):
        return [0.1] * self.get_vector_size()

@pytest.mark.asyncio
async def test_store_entry():
    connector = QdrantConnector(
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        collection_name="test-memory",
        embedding_provider=MockEmbeddingProvider(),
        qdrant_local_path=None,
    )

    entry = Entry(
        content="Este es un contenido de prueba para almacenar en Qdrant.",
        payload={"name": "Arturo", "role": "developer"},
        metadata={"source": "pytest"}
    )

    await connector.store(entry)

