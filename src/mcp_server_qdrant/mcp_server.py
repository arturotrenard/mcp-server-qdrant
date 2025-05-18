import json
import logging
from typing import Any, List

from mcp.server.fastmcp import Context, FastMCP

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: EmbeddingProviderSettings,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
        self.embedding_provider_settings = embedding_provider_settings

        self.embedding_provider = create_embedding_provider(embedding_provider_settings)
        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        parts = [f"{k}: {v}" for k, v in entry.payload.items()]
        return " | ".join(parts)

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
                ctx: Context,
                information: str,
                collection_name: str,
                metadata: Metadata = None,
        ) -> str:
            await ctx.debug(f"Storing information {information} in Qdrant")

            payload = {
                "content": information,
            }

            entry = Entry(payload=payload)
            await self.qdrant_connector.store(entry, collection_name=collection_name)

            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def store_with_default_collection(
            ctx: Context,
            information: str,
            metadata: Metadata = None,  # type: ignore
        ) -> str:
            assert self.qdrant_settings.collection_name is not None
            return await store(
                ctx, information, self.qdrant_settings.collection_name, metadata
            )

        async def find(
            ctx: Context,
            query: str,
            collection_name: str,
        ) -> List[str]:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :return: A list of entries found.
            """
            await ctx.debug(f"Finding results for query {query}")
            if collection_name:
                await ctx.debug(
                    f"Overriding the collection name with {collection_name}"
                )

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
            )
            if not entries:
                return [f"No information found for the query '{query}'"]
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        async def find_recent(
                ctx: Context,
                query: str,
                # opcionales ↓
                days: int | None = None,
                after_ts: int | None = None,
                limit: int | None = None,
                collection_name: str | None = None,
        ) -> list[dict]:
            """
            Igual que qdrant-find, pero filtra por fecha.
            - `days`: últimos N días.
            - `after_ts`: epoch-ms.  Si se dan ambos, after_ts tiene prioridad.
            - `limit`: cuántos resultados (por defecto QDRANT_SEARCH_LIMIT).
            """
            await ctx.debug(
                f"find_recent query='{query}' after_ts={after_ts} days={days}"
            )

            coll = (collection_name or self.qdrant_settings.collection_name)
            lim = limit or self.qdrant_settings.search_limit

            entries = await self.qdrant_connector.search_recent(
                query,
                collection_name=coll,
                limit=lim,
                days=days,
                after_ts=after_ts,
            )
            if not entries:
                return [{"message": f"No recent information for '{query}'"}]

            return [e.payload for e in entries]

        async def find_with_default_collection(
            ctx: Context,
            query: str,
        ) -> List[str]:
            assert self.qdrant_settings.collection_name is not None
            return await find(ctx, query, self.qdrant_settings.collection_name)

        # Register the tools depending on the configuration

        if self.qdrant_settings.collection_name:
            self.add_tool(
                find_with_default_collection,
                name="qdrant-find",
                description=self.tool_settings.tool_find_description,
            )
        else:
            self.add_tool(
                find,
                name="qdrant-find",
                description=self.tool_settings.tool_find_description,
            )

        self.add_tool(
            find_recent,
            name="qdrant-find-recent",
            description=(
                "Look up memories in Qdrant that match `query` and were "
                "published after `after_ts` (epoch-ms) or within the last "
                "`days` days. Returns newest first with full payload "
                "(title, source_url, published_date, tags…)."
            ),
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database

            if self.qdrant_settings.collection_name:
                self.add_tool(
                    store_with_default_collection,
                    name="qdrant-store",
                    description=self.tool_settings.tool_store_description,
                )
            else:
                self.add_tool(
                    store,
                    name="qdrant-store",
                    description=self.tool_settings.tool_store_description,
                )
