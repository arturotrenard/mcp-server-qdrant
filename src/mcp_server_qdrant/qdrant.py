import json
import logging
import uuid
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

Metadata = Dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """
    payload: Dict[str, Any]


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
        collection_name: Optional[str],
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: Optional[str] = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: Optional[str] = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        :raises ValueError: If `entry.content` is None or empty, or if the collection name is invalid.
        """
        try:
            # Validate collection name
            collection_name = collection_name or self._default_collection_name
            if not collection_name:
                raise ValueError("No collection name provided and no default collection set.")
            logger.info(f"Storing entry in collection: {collection_name}")

            # Validate entry content
            if not entry.payload:
                raise ValueError("Entry payload cannot be None or empty.")
            logger.debug(f"Entry payload preview: {json.dumps(entry.payload)[:200]}...")   # Log first 100 chars to avoid excessive logs

            # Ensure collection exists
            await self._ensure_collection_exists(collection_name)
            logger.debug(f"Collection {collection_name} exists or was created successfully.")

            content = entry.payload.get("content")
            if not content:
                raise ValueError("Missing 'content' in payload to generate embeddings.")

            # Embed the document
            embeddings = await self._embedding_provider.embed_documents([content])
            if not embeddings or not embeddings[0]:
                raise ValueError("Failed to generate embeddings for the entry content.")
            logger.debug("Embeddings generated successfully.")

            # Prepare payload and vector field
            vector_name = self._embedding_provider.get_vector_name()
            payload = entry.payload
            vec_field = (
                embeddings[0]
                if vector_name in ("", None)
                else {vector_name: embeddings[0]}
            )
            logger.debug(f"Vector field prepared with vector name: {vec_field}")

            # Upsert into Qdrant
            point_id = uuid.uuid4().hex
            await self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vec_field,
                        payload=payload,
                    )
                ],
            )
            logger.info(f"Entry stored successfully with ID: {point_id}")

        except Exception as e:
            logger.error(f"Failed to store entry: {e}", exc_info=True)
            raise  # Re-raise the exception for the caller to handle

    async def search(
            self, query: str, *, collection_name: Optional[str] = None, limit: int = 10
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        vec_name = None if not vector_name else vector_name

        logger.debug(f"Vector field prepared with vector name: {vec_name}")

        # Search in Qdrant
        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vec_name,
            limit=limit,
            with_payload=True,
        )

        return [
            Entry(payload=p.payload)
            for p in search_results.points
        ]

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the vector name as defined in the embedding provider
            vector_name = self._embedding_provider.get_vector_name()
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )

    async def search_recent(
            self,
            query: str,
            *,
            limit: int = 5,
            days: int | None = None,  # últimos N días
            after_ts: int | None = None,  # timestamp (ms)
            collection_name: str | None = None,
    ):
        if after_ts is None and days is not None:
            after_ts = int(
                (datetime.utcnow() - timedelta(days=days)).timestamp() * 1000
            )

        filt = None
        if after_ts:
            filt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="published_date",
                        range=models.Range(gte=after_ts),
                    )
                ]
            )

        entries = await self._search_with_filter(
            query=query,
            flt=filt,
            limit=limit,
            coll=collection_name,
        )

        # ordena de más nuevo a más viejo
        entries.sort(
            key=lambda e: e.payload.get("published_date", 0),
            reverse=True,
        )
        return entries

    async def _search_with_filter(
            self, query: str, coll: str | None, limit: int, flt: models.Filter
    ) -> list[Entry]:

        vec = await self._embedding_provider.embed_query(query)
        vname = self._embedding_provider.get_vector_name() or None

        logger.debug(f"Vector field prepared with vector name: {vname}")

        res = await self._client.query_points(
            collection_name=coll,
            query=vec,
            using=vname,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )

        return [
            Entry(payload=p.payload)
            for p in res.points
        ]
