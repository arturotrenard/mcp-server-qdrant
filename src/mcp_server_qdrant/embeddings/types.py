from enum import Enum


class EmbeddingProviderType(str, Enum):
    FASTEMBED = "fastembed"
    OLLAMA = "ollama"