# Dockerfile
FROM python:3.11-slim
WORKDIR /app

# 1. Copia tu fork completo
COPY . /app

# 2. Instala uv y tus dependencias **en editable**
RUN pip install --no-cache-dir uv \
 && uv pip install --system --no-cache-dir -e .

# 3. Variables por defecto (se pueden sobreescribir al ejecutar)
ENV QDRANT_URL="http://192.168.10.3:32726" \
    QDRANT_API_KEY="" \
    COLLECTION_NAME="news" \
    EMBEDDING_MODEL="bge-m3" \
    EMBEDDING_PROVIDER="ollama" \
    OLLAMA_BASE_URL="http://192.168.10.3:31784"

EXPOSE 8000
CMD ["mcp-server-qdrant", "--transport", "sse"]
