"""Dependencies for PostgreSQL RAG Agent."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging
import asyncpg
import openai

from src.settings import load_settings
from src.ingestion.embedder import resolve_embedding_dimension

logger = logging.getLogger(__name__)


@dataclass
class AgentDependencies:
    """Dependencies injected into the agent context."""

    # Core dependencies
    db_pool: Optional[asyncpg.Pool] = None
    openai_client: Optional[openai.AsyncOpenAI] = None
    settings: Optional[Any] = None

    # Session context
    session_id: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    query_history: list = field(default_factory=list)

    async def initialize(self) -> None:
        """
        Initialize external connections.
        """
        if not self.settings:
            self.settings = load_settings()
            logger.info("settings_loaded", database=self.settings.database_url)

        if not self.db_pool:
            self.db_pool = await asyncpg.create_pool(
                self.settings.database_url,
                min_size=self.settings.db_pool_min_size,
                max_size=self.settings.db_pool_max_size,
                command_timeout=10,
            )
            async with self.db_pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            logger.info("postgres_pool_initialized")

        if not self.openai_client:
            self.openai_client = openai.AsyncOpenAI(
                api_key=self.settings.embedding_api_key,
                base_url=self.settings.embedding_base_url,
            )
            logger.info(
                "openai_client_initialized",
                model=self.settings.embedding_model,
            )

    async def cleanup(self) -> None:
        """Clean up external connections."""
        if self.db_pool:
            await self.db_pool.close()
            self.db_pool = None
            logger.info("postgres_pool_closed")

    async def get_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for text using OpenAI.
        """
        if not self.openai_client:
            await self.initialize()

        dim = resolve_embedding_dimension(
            self.settings.embedding_model,
            self.settings.embedding_dimension,
        )

        response = await self.openai_client.embeddings.create(
            model=self.settings.embedding_model,
            input=text,
            dimensions=dim,
        )
        return response.data[0].embedding

    def set_user_preference(self, key: str, value: Any) -> None:
        """Set a user preference for the session."""
        self.user_preferences[key] = value

    def add_to_history(self, query: str) -> None:
        """Add a query to the search history."""
        self.query_history.append(query)
        if len(self.query_history) > 10:
            self.query_history.pop(0)
