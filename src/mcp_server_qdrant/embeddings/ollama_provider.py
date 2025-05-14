# src/mcp_server_qdrant/embeddings/ollama_provider.py
from typing import List
from ollama import AsyncClient
from mcp_server_qdrant.embeddings.base import EmbeddingProvider

VECTOR_DIM = 1024        # bge-m3
DEFAULT_BASE = "http://192.168.10.3:31784"

class OllamaEmbedProvider(EmbeddingProvider):
    def __init__(self, model_name: str,
                 base_url: str | None = None,
                 timeout: int = 120):
        self.model_name = model_name
        self.client = AsyncClient(
            host=(base_url or DEFAULT_BASE).rstrip("/"),
            timeout=timeout,
        )

    async def _embed(self, text: str) -> List[float]:
        r = await self.client.embed(model=self.model_name, input=text)
        return r["embeddings"][0]

    async def embed_documents(self, docs: List[str]) -> List[List[float]]:
        from asyncio import gather
        return await gather(*[self._embed(d) for d in docs])

    async def embed_query(self, query: str) -> List[float]:
        return await self._embed(query)

    def get_vector_name(self) -> str:
        return ""

    def get_vector_size(self) -> int:
        return VECTOR_DIM

