"""
Ingestion pipeline for PostgreSQL + pgvector.

Processes documents, chunks them, embeds with configured model, and writes to Postgres
with vector and full-text indexes.
"""

import os
import asyncio
import logging
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse
from dataclasses import dataclass
import json

import asyncpg
from dotenv import load_dotenv

from src.ingestion.chunker import ChunkingConfig, create_chunker, DocumentChunk
from src.ingestion.embedder import create_embedder, resolve_embedding_dimension
from src.settings import load_settings

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for document ingestion."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    max_tokens: int = 512


@dataclass
class IngestionResult:
    """Result of document ingestion."""

    document_id: str
    title: str
    chunks_created: int
    processing_time_ms: float
    errors: List[str]


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into PostgreSQL with pgvector."""

    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "documents",
        clean_before_ingest: bool = True,
    ):
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest

        self.settings = load_settings()
        self.pool: Optional[asyncpg.Pool] = None

        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            max_tokens=config.max_tokens,
        )
        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()
        self.embedding_dimension = resolve_embedding_dimension(
            self.settings.embedding_model, self.settings.embedding_dimension
        )

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Postgres connections and schema."""
        if self._initialized:
            return

        logger.info("Initializing ingestion pipeline...")
        self.pool = await asyncpg.create_pool(
            self.settings.database_url,
            min_size=self.settings.db_pool_min_size,
            max_size=self.settings.db_pool_max_size,
            command_timeout=10,
        )

        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            await self._ensure_schema(conn)

        self._initialized = True
        logger.info("Ingestion pipeline initialized")

    async def close(self) -> None:
        """Close Postgres connections."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False
            logger.info("Postgres pool closed")

    async def ingest_documents(
        self, progress_callback: Optional[callable] = None
    ) -> List[IngestionResult]:
        """Ingest all supported documents in the documents folder."""
        if not self._initialized:
            await self.initialize()

        if self.clean_before_ingest:
            await self._clean_database()

        document_files = self._find_document_files()
        if not document_files:
            logger.warning(f"No supported document files found in {self.documents_folder}")
            return []

        results: List[IngestionResult] = []
        for i, file_path in enumerate(document_files):
            try:
                result = await self._ingest_single_document(file_path)
                results.append(result)
            except Exception as e:
                logger.exception(f"Failed to ingest {file_path}")
                results.append(
                    IngestionResult(
                        document_id="",
                        title=os.path.basename(file_path),
                        chunks_created=0,
                        processing_time_ms=0,
                        errors=[str(e)],
                    )
                )

            if progress_callback:
                progress_callback(i + 1, len(document_files))

        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results)
        logger.info(
            f"Ingestion complete: {len(results)} documents, {total_chunks} chunks, {total_errors} errors"
        )
        return results

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        start = datetime.now()

        content, docling_doc = self._read_document(file_path)
        title = self._extract_title(content, file_path)
        source = os.path.relpath(file_path, self.documents_folder)
        metadata = self._extract_document_metadata(content, file_path)

        logger.info(f"Chunking {title}")
        chunks = await self.chunker.chunk_document(
            content=content,
            title=title,
            source=source,
            metadata=metadata,
            docling_doc=docling_doc,
        )

        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embedded_chunks = await self.embedder.embed_chunks(chunks)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                document_id = await conn.fetchval(
                    """
                    INSERT INTO documents (title, source, metadata, content)
                    VALUES ($1, $2, $3::jsonb, $4)
                    RETURNING id;
                    """,
                    title,
                    source,
                    json.dumps(metadata),
                    content,
                )

                chunk_rows = [
                    (
                        document_id,
                        chunk.content,
                        json.dumps(chunk.metadata),
                        "[" + ",".join(map(str, chunk.embedding)) + "]",
                        chunk.token_count,
                        chunk.index,
                    )
                    for chunk in embedded_chunks
                ]

                await conn.executemany(
                    """
                    INSERT INTO chunks (
                        document_id,
                        content,
                        metadata,
                        embedding,
                        token_count,
                        chunk_index
                    ) VALUES ($1, $2, $3::jsonb, $4::vector, $5, $6);
                    """,
                    chunk_rows,
                )

        elapsed_ms = (datetime.now() - start).total_seconds() * 1000
        logger.info(f"Ingested {title} with {len(embedded_chunks)} chunks")
        return IngestionResult(
            document_id=str(document_id),
            title=title,
            chunks_created=len(embedded_chunks),
            processing_time_ms=elapsed_ms,
            errors=[],
        )

    async def _clean_database(self) -> None:
        logger.warning("Cleaning existing data from Postgres tables...")
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("TRUNCATE TABLE chunks RESTART IDENTITY CASCADE;")
                await conn.execute("TRUNCATE TABLE documents RESTART IDENTITY CASCADE;")

    async def _ensure_schema(self, conn: asyncpg.Connection) -> None:
        """Ensure tables and indexes exist."""
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                title TEXT NOT NULL,
                source TEXT,
                content TEXT,
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMPTZ DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{{}}',
                embedding vector({self.embedding_dimension}) NOT NULL,
                token_count INTEGER,
                chunk_index INTEGER,
                created_at TIMESTAMPTZ DEFAULT now()
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

            CREATE INDEX IF NOT EXISTS idx_chunks_fts
                ON chunks USING GIN (to_tsvector('english', content));
            """
        )

    def _find_document_files(self) -> List[str]:
        """Find all supported document files in the documents folder."""
        if not os.path.exists(self.documents_folder):
            logger.error(f"Documents folder not found: {self.documents_folder}")
            return []

        patterns = [
            "*.md",
            "*.markdown",
            "*.txt",
            "*.pdf",
            "*.docx",
            "*.doc",
            "*.pptx",
            "*.ppt",
            "*.xlsx",
            "*.xls",
            "*.html",
            "*.htm",
            "*.mp3",
            "*.wav",
            "*.m4a",
            "*.flac",
        ]
        files: List[str] = []
        for pattern in patterns:
            files.extend(
                glob.glob(
                    os.path.join(self.documents_folder, "**", pattern),
                    recursive=True,
                )
            )
        return sorted(files)

    def _read_document(self, file_path: str) -> tuple[str, Optional[Any]]:
        """Read document content using Docling where possible."""
        file_ext = os.path.splitext(file_path)[1].lower()
        audio_formats = [".mp3", ".wav", ".m4a", ".flac"]
        docling_formats = [
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".xlsx",
            ".xls",
            ".html",
            ".htm",
            ".md",
            ".markdown",
        ]

        if file_ext in audio_formats:
            return self._transcribe_audio(file_path)

        if file_ext in docling_formats:
            try:
                from docling.document_converter import DocumentConverter

                converter = DocumentConverter()
                result = converter.convert(file_path)
                markdown_content = result.document.export_to_markdown()
                return (markdown_content, result.document)
            except Exception as e:
                logger.error(f"Docling failed for {file_path}: {e}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return (f.read(), None)
                except Exception:
                    return (f"[Error reading file {os.path.basename(file_path)}]", None)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return (f.read(), None)
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                return (f.read(), None)

    def _transcribe_audio(self, file_path: str) -> tuple[str, Optional[Any]]:
        """Placeholder for audio transcription; currently unsupported inline."""
        return (f"[Audio transcription not implemented for {file_path}]", None)

    def _extract_title(self, content: str, file_path: str) -> str:
        """Extract title from markdown content or fallback to filename."""
        for line in content.splitlines():
            if line.strip().startswith("#"):
                return line.strip("# ").strip()
        return Path(file_path).stem

    def _extract_document_metadata(
        self,
        content: str,
        file_path: str,
        frontmatter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        if frontmatter:
            metadata.update(frontmatter)

        lines = content.split("\n")
        metadata["line_count"] = len(lines)
        metadata["word_count"] = len(content.split())
        metadata["filename"] = os.path.basename(file_path)

        return metadata


async def main(args: argparse.Namespace) -> None:
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_chunk_size=args.max_chunk_size,
        max_tokens=args.max_tokens,
    )

    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=not args.no_clean,
    )

    try:
        await pipeline.initialize()
        await pipeline.ingest_documents()
    finally:
        await pipeline.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Postgres + pgvector")
    parser.add_argument(
        "-d",
        "--documents",
        type=str,
        default="documents",
        help="Path to documents folder",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for splitting documents",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=2000,
        help="Maximum chunk size",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk for safety trimming",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not wipe existing tables before ingest",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
