from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.embeddings.ollama_provider import OllamaEmbedProvider
import os


def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    :param settings: The settings for the embedding provider.
    :return: An instance of the specified embedding provider.
    """
    if settings.provider_type == EmbeddingProviderType.FASTEMBED:
        return FastEmbedProvider(settings.model_name)
    if settings.provider_type is EmbeddingProviderType.OLLAMA:
        return OllamaEmbedProvider(
            model_name=settings.model_name,
            base_url=os.getenv("OLLAMA_BASE_URL"),
        )
    raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")
