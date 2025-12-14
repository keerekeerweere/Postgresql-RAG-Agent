"""
Document embedding generation for vector search.
"""

import logging
from typing import List, Optional
from datetime import datetime

from dotenv import load_dotenv
import openai

from src.ingestion.chunker import DocumentChunk
from src.settings import load_settings

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize client with settings
settings = load_settings()
embedding_client = openai.AsyncOpenAI(
    api_key=settings.embedding_api_key,
    base_url=settings.embedding_base_url
)
EMBEDDING_MODEL = settings.embedding_model


class EmbeddingGenerator:
    """Generates embeddings for document chunks."""

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = 100
    ):
        """
        Initialize embedding generator.

        Args:
            model: Embedding model to use
            batch_size: Number of texts to process in parallel
        """
        self.model = model
        self.batch_size = batch_size

        # Model-specific configurations
        self.model_configs = {
            "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
            "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
            "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191},
            "qwen3-embedding": {"dimensions": 1536, "max_tokens": 8191},
            "mxbai-embed-large": {"dimensions": 1024, "max_tokens": 8191}
        }

        self.config = self.model_configs.get(
            model,
            {"dimensions": 1536, "max_tokens": 8191}
        )

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Truncate text if too long (rough estimation: 4 chars per token)
        if len(text) > self.config["max_tokens"] * 4:
            text = text[:self.config["max_tokens"] * 4]

        response = await embedding_client.embeddings.create(
            model=self.model,
            input=text
        )

        return response.data[0].embedding

    async def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Truncate texts if too long
        processed_texts = []
        for text in texts:
            if len(text) > self.config["max_tokens"] * 4:
                text = text[:self.config["max_tokens"] * 4]
            processed_texts.append(text)

        response = await embedding_client.embeddings.create(
            model=self.model,
            input=processed_texts
        )

        return [data.embedding for data in response.data]

    async def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        progress_callback: Optional[callable] = None
    ) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of document chunks
            progress_callback: Optional callback for progress updates

        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            return chunks

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Process chunks in batches
        embedded_chunks = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]

            # Generate embeddings for this batch
            embeddings = await self.generate_embeddings_batch(batch_texts)

            # Add embeddings to chunks
            for chunk, embedding in zip(batch_chunks, embeddings):
                embedded_chunk = DocumentChunk(
                    content=chunk.content,
                    index=chunk.index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={
                        **chunk.metadata,
                        "embedding_model": self.model,
                        "embedding_generated_at": datetime.now().isoformat()
                    },
                    token_count=chunk.token_count
                )
                embedded_chunk.embedding = embedding
                embedded_chunks.append(embedded_chunk)

            # Progress update
            current_batch = (i // self.batch_size) + 1
            if progress_callback:
                progress_callback(current_batch, total_batches)

            logger.info(f"Processed batch {current_batch}/{total_batches}")

        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.

        Args:
            query: Search query

        Returns:
            Query embedding
        """
        return await self.generate_embedding(query)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        return self.config["dimensions"]


def create_embedder(model: str = EMBEDDING_MODEL, **kwargs) -> EmbeddingGenerator:
    """
    Create embedding generator.

    Args:
        model: Embedding model to use
        **kwargs: Additional arguments for EmbeddingGenerator

    Returns:
        EmbeddingGenerator instance
    """
    return EmbeddingGenerator(model=model, **kwargs)
