# Feature: Phase 2 - Document Ingestion Pipeline

## Feature Description

Build a document ingestion pipeline that processes multi-format documents (PDF, Word, PowerPoint, Excel, Markdown, Audio) into searchable chunks with embeddings stored in MongoDB Atlas. Adapts the PostgreSQL pipeline from `examples/ingestion/` to MongoDB, changing only the database layer while preserving all document processing logic.

## User Story

As a **developer setting up the MongoDB RAG knowledge base**
I want to **ingest documents from multiple formats into MongoDB with proper chunking and embeddings**
So that **I have a searchable knowledge base ready for the AI agent to query**

## Problem Statement

Need to:
1. Create MongoDB async connection management
2. Adapt ingestion pipeline to MongoDB insert operations
3. Handle embeddings as Python lists (not pgvector strings)
4. Preserve two-collection pattern (documents + chunks with document_id FKs)
5. Support 10+ document formats via Docling
6. Provide CLI with progress tracking

## Solution Statement

Create `src/dependencies.py` for MongoDB connections and `src/ingestion/` with:
- **chunker.py**: Copy from examples (platform-agnostic)
- **embedder.py**: Copy from examples with updated imports
- **ingest.py**: Adapt from examples, replace PostgreSQL → MongoDB async API

Pipeline: MongoDB connect → Discover docs → Docling convert → HybridChunk → Batch embeddings → Two-collection insert → Progress tracking

## Feature Metadata

**Feature Type**: New Capability (Core Infrastructure)
**Estimated Complexity**: Medium
**Primary Systems Affected**: Database Layer, Document Processing, Embedding Generation
**Dependencies**: PyMongo 4.10+, Docling 2.14+, OpenAI 1.58+, Transformers 4.47+, Rich 13.9+

---

## CONTEXT REFERENCES

### Relevant Codebase Files (READ BEFORE IMPLEMENTING!)

**Examples (PostgreSQL Reference):**
- `examples/ingestion/chunker.py` (all) - Docling HybridChunker, copy as-is
- `examples/ingestion/embedder.py` (all) - Batch embeddings, copy with import changes
- `examples/ingestion/ingest.py` (1-550) - Orchestration, adapt PostgreSQL→MongoDB
- `examples/dependencies.py` (1-71) - Dependency injection pattern

**Existing Src (Phase 1):**
- `src/settings.py` (1-105) - MongoDB config
- `src/providers.py` (1-81) - LLM/embedding clients

**Reference Docs:**
- `.claude/reference/mongodb-patterns.md` (1-514) - MongoDB async, two-collection design
- `.claude/reference/docling-ingestion.md` (1-534) - Document conversion patterns
- `CLAUDE.md` (1-300) - TYPE SAFETY, ASYNC, Google docstrings

### New Files to Create

- `src/dependencies.py` - MongoDB AsyncMongoClient connection management
- `src/ingestion/__init__.py` - Module init
- `src/ingestion/chunker.py` - Copy from examples
- `src/ingestion/embedder.py` - Copy from examples, update imports
- `src/ingestion/ingest.py` - Adapt from examples for MongoDB

### Key Documentation

- [PyMongo Async API](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/get-started/quickstart/) - AsyncMongoClient patterns
- [MongoDB Vector RAG Guide](https://www.mongodb.com/docs/atlas/atlas-vector-search/rag/) - Data ingestion with embeddings
- [Docling Chunking](https://docling.dev/usage/chunking/) - HybridChunker usage

### Critical Patterns

**MongoDB Connection:**
```python
# From .claude/reference/mongodb-patterns.md
from pymongo import AsyncMongoClient
from pymongo.errors import ConnectionFailure

self.mongo_client = AsyncMongoClient(settings.mongodb_uri)
self.db = self.mongo_client[settings.mongodb_database]
await self.mongo_client.admin.command('ping')  # Verify
```

**Two-Collection Insert:**
```python
# Insert document, get ObjectId
doc_result = await db.documents.insert_one({
    "title": title, "source": source, "content": content,
    "metadata": metadata, "created_at": datetime.utcnow()
})
document_id = doc_result.inserted_id

# Insert chunks with embedding as list
chunk_dicts = [{
    "document_id": document_id,
    "content": chunk.content,
    "embedding": chunk.embedding,  # Python list!
    "chunk_index": chunk.index,
    "metadata": chunk.metadata,
    "token_count": chunk.token_count
} for chunk in chunks]
await db.chunks.insert_many(chunk_dicts, ordered=False)
```

**Docling Conversion:**
```python
converter = DocumentConverter()
result = converter.convert(file_path)
markdown = result.document.export_to_markdown()
docling_doc = result.document  # Keep for HybridChunker!
```

---

## IMPLEMENTATION PLAN

### Phase 1: MongoDB Dependencies
Create `src/dependencies.py` with AsyncMongoClient, lazy init, cleanup, connection verification

### Phase 2: Ingestion Components
Copy chunker/embedder from examples (platform-agnostic), update import paths to `src.*`

### Phase 3: MongoDB Pipeline
Adapt `ingest.py` - replace asyncpg with PyMongo, embeddings as lists, two-collection pattern

### Phase 4: CLI & Validation
Add argparse CLI, progress callbacks, summary stats, end-to-end testing

---

## STEP-BY-STEP TASKS

### CREATE src/dependencies.py

- **IMPLEMENT**: MongoDB connection with PyMongo AsyncMongoClient
- **PATTERN**: examples/dependencies.py structure + .claude/reference/mongodb-patterns.md (191-250)
- **IMPORTS**: `dataclass, AsyncMongoClient, ConnectionFailure, ServerSelectionTimeoutError, openai, load_settings, logging`
- **KEY METHODS**:
  - `async def initialize()`: Create AsyncMongoClient, get db ref, ping to verify
  - `async def cleanup()`: Close client, null references
  - `async def get_embedding(text)`: Generate single embedding via OpenAI
- **GOTCHA**: Use `AsyncMongoClient` from pymongo (NOT motor's AsyncIOMotorClient)
- **GOTCHA**: Verify with `await client.admin.command('ping')` - no built-in test
- **GOTCHA**: Embeddings as `list[float]`, NOT string `'[0.1,0.2,...]'`
- **VALIDATE**: `uv run python -c "import asyncio; from src.dependencies import AgentDependencies; asyncio.run(AgentDependencies().initialize())"`

### CREATE src/ingestion/__init__.py

- **IMPLEMENT**: Module init with docstring
- **CONTENT**: `"""Document ingestion pipeline for MongoDB RAG Agent."""`
- **VALIDATE**: `uv run python -c "import src.ingestion; print('Module imported')"`

### COPY examples/ingestion/chunker.py → src/ingestion/chunker.py

- **IMPLEMENT**: Copy entire file unchanged (platform-agnostic)
- **GOTCHA**: File is 100% reusable - no database dependencies
- **GOTCHA**: HybridChunker needs `DoclingDocument` object, not string
- **VALIDATE**: `uv run python -c "from src.ingestion.chunker import ChunkingConfig, DoclingHybridChunker; print('Chunker imported')"`

### COPY examples/ingestion/embedder.py → src/ingestion/embedder.py

- **IMPLEMENT**: Copy file, update imports to use `src.providers` and `src.settings`
- **CHANGES**:
  - Replace `..utils.providers` imports with `src.providers`
  - Initialize embedding_client using `src.settings.load_settings()`
  - Keep all class methods unchanged
- **GOTCHA**: Import from `src.providers`, not relative imports
- **VALIDATE**: `uv run python -c "from src.ingestion.embedder import EmbeddingGenerator, create_embedder; print('Embedder imported')"`

### CREATE src/ingestion/ingest.py - Part 1: Skeleton

- **IMPLEMENT**: Imports, dataclasses (IngestionConfig, IngestionResult), DocumentIngestionPipeline class init
- **IMPORTS**: `asyncio, logging, glob, pathlib, pymongo.AsyncMongoClient, pymongo.errors, bson.ObjectId, datetime, argparse`
- **CLASS INIT**: Store config, folders, chunker, embedder, settings, mongo client/db refs, _initialized flag
- **METHODS**: `async def initialize()`, `async def close()`, `_find_document_files()`
- **GOTCHA**: Import `AsyncMongoClient` from pymongo, `ObjectId` from bson
- **VALIDATE**: `uv run python -c "from src.ingestion.ingest import IngestionConfig, IngestionResult; print('Config imported')"`

### CREATE src/ingestion/ingest.py - Part 2: Document Processing

- **IMPLEMENT**: `_read_document()`, `_extract_title()`, `_extract_document_metadata()`
- **PATTERN**: .claude/reference/docling-ingestion.md (9-52, 390-448)
- **KEY LOGIC**:
  - `_read_document()`: Docling convert, return `(markdown, docling_doc)` tuple, fallback to raw text
  - `_extract_title()`: Parse markdown `# Title` or use filename
  - `_extract_document_metadata()`: File stats, YAML frontmatter extraction
- **GOTCHA**: MUST return `docling_doc` (DoclingDocument object) for HybridChunker
- **GOTCHA**: Handle encoding errors with latin-1 fallback
- **VALIDATE**: Test after full implementation

### CREATE src/ingestion/ingest.py - Part 3: MongoDB Operations

- **IMPLEMENT**: `_save_to_mongodb()`, `_clean_databases()`
- **PATTERN**: .claude/reference/mongodb-patterns.md (436-475)
- **SAVE LOGIC**:
  - Insert document → get `inserted_id` (ObjectId)
  - Build chunk dicts with `document_id`, `embedding` (list!), metadata
  - `insert_many(chunk_dicts, ordered=False)` for batch insert
- **CLEAN LOGIC**: `delete_many({})` on chunks, then documents
- **GOTCHA**: Embeddings MUST be `list[float]`, NOT string
- **GOTCHA**: `ordered=False` allows partial success
- **VALIDATE**: Test after full implementation

### CREATE src/ingestion/ingest.py - Part 4: Pipeline Methods

- **IMPLEMENT**: `_ingest_single_document()`, `ingest_documents()`
- **PATTERN**: examples/ingestion/ingest.py (172-250, 111-170)
- **SINGLE DOC**: Read → chunk (pass docling_doc!) → embed → save → return IngestionResult
- **BATCH DOCS**: Find files → clean if requested → loop with try/except → progress callbacks → summary
- **GOTCHA**: Pass `docling_doc` to chunker, not just string content
- **GOTCHA**: Continue on errors (one bad file doesn't stop pipeline)
- **VALIDATE**: Test after CLI implementation

### CREATE src/ingestion/ingest.py - Part 5: CLI Main

- **IMPLEMENT**: `async def main()` with argparse, logging, pipeline execution, summary
- **ARGS**: `--documents/-d`, `--no-clean`, `--chunk-size`, `--chunk-overlap`, `--max-tokens`, `--verbose/-v`
- **FLOW**: Parse args → config logging → create pipeline → initialize → ingest with progress → print summary → cleanup
- **SUMMARY**: Show doc count, chunks created, errors, time, per-doc results, next steps (create indexes)
- **GOTCHA**: Use `asyncio.run(main())` at bottom
- **GOTCHA**: Always cleanup in finally block
- **VALIDATE**: `uv run python -m src.ingestion.ingest --help`

---

## TESTING STRATEGY

### Unit Tests (Phase 5)
- Mock MongoDB client with pytest-asyncio
- Mock OpenAI API for embedder tests
- Test chunker config validation, metadata extraction
- Use sample fixtures in `tests/fixtures/`

### Integration Tests
- Use test MongoDB database (not production)
- Ingest sample documents, verify counts
- Validate embedding dimensions (1536)
- Check document-chunk relationships

### Edge Cases
1. Empty documents - skip with warning
2. Huge docs (>100K words) - chunk without memory issues
3. Invalid UTF-8 - fallback to latin-1
4. Docling failure - fallback to raw text
5. MongoDB disconnect - fail gracefully
6. Missing API key - fail fast with clear error

---

## VALIDATION COMMANDS

### Level 1: Imports
```bash
uv run python -c "from src.dependencies import AgentDependencies; print('✓ deps')"
uv run python -c "from src.ingestion.chunker import DoclingHybridChunker; print('✓ chunker')"
uv run python -c "from src.ingestion.embedder import EmbeddingGenerator; print('✓ embedder')"
uv run python -c "from src.ingestion.ingest import DocumentIngestionPipeline; print('✓ ingest')"
```

### Level 2: MongoDB Connection
```bash
uv run python -c "import asyncio; from src.dependencies import AgentDependencies; asyncio.run(AgentDependencies().initialize()); print('✓ connected')"
```

### Level 3: Component Testing
```bash
uv run python -c "from src.ingestion.chunker import ChunkingConfig, create_chunker; chunker = create_chunker(ChunkingConfig()); print('✓ chunker')"
uv run python -c "from src.ingestion.embedder import create_embedder; e = create_embedder(); print(f'✓ embedder: {e.get_embedding_dimension()} dims')"
uv run python -c "from src.ingestion.ingest import DocumentIngestionPipeline, IngestionConfig; p = DocumentIngestionPipeline(IngestionConfig()); files = p._find_document_files(); print(f'✓ found {len(files)} docs')"
```

### Level 4: End-to-End
```bash
uv run python -m src.ingestion.ingest --help
uv run python -m src.ingestion.ingest -d documents --verbose
uv run python -m src.ingestion.ingest -d documents
```

### Level 5: Data Integrity
```bash
# Verify embeddings are lists
uv run python -c "import asyncio; from src.dependencies import AgentDependencies; async def test(): deps = AgentDependencies(); await deps.initialize(); chunk = await deps.db.chunks.find_one(); assert isinstance(chunk['embedding'], list); assert len(chunk['embedding']) == 1536; print('✓ embeddings valid'); await deps.cleanup(); asyncio.run(test())"

# Verify document-chunk relationships
uv run python -c "import asyncio; from src.dependencies import AgentDependencies; async def test(): deps = AgentDependencies(); await deps.initialize(); doc = await deps.db.documents.find_one(); count = await deps.db.chunks.count_documents({'document_id': doc['_id']}); print(f'✓ doc has {count} chunks'); await deps.cleanup(); asyncio.run(test())"
```

---

## ACCEPTANCE CRITERIA

- [ ] `src/dependencies.py` with MongoDB AsyncMongoClient connection
- [ ] MongoDB connection verifies successfully with ping
- [ ] `src/ingestion/chunker.py` copied from examples
- [ ] `src/ingestion/embedder.py` copied with src.providers imports
- [ ] `src/ingestion/ingest.py` with MongoDB async operations
- [ ] Document discovery finds 10+ format extensions
- [ ] Docling conversion works for PDF, Word, PowerPoint, Excel
- [ ] HybridChunker receives DoclingDocument (not string)
- [ ] Embeddings generated in batches (100/batch)
- [ ] Embeddings stored as Python lists (NOT strings)
- [ ] Two-collection pattern works (documents + chunks with FK)
- [ ] CLI accepts all flags (--documents, --no-clean, --verbose, etc.)
- [ ] Progress callback shows real-time updates
- [ ] Summary stats printed after completion
- [ ] Error handling continues on individual failures
- [ ] All validation commands pass
- [ ] MongoDB populated after successful ingestion
- [ ] Data integrity validated (embeddings as lists, proper FKs)

---

## NOTES

### Design Decisions

**PyMongo AsyncMongoClient**: Modern standard for MongoDB async in Python (2025), replaces Motor

**Two-Collection Pattern**: Documents (source) + Chunks (searchable) maintains normalization, enables $lookup joins

**Copy Don't Modify**: examples/ stays pristine as PostgreSQL reference

**Embeddings as Lists**: MongoDB native arrays - MUST be `list[float]`, NOT pgvector string `'[...]'`

### Critical Dependencies

1. **Atlas Indexes**: MUST create in Atlas UI AFTER ingestion (vector_index on embedding, text_index on content). See .claude/reference/mongodb-patterns.md lines 254-321
2. **DoclingDocument Object**: HybridChunker needs object from converter, not markdown string
3. **Batch Size**: 100 chunks/batch balances API limits and performance

### Security

- MongoDB URI contains credentials - ensure .env gitignored
- API keys in .env, never in code
- Use `ordered=False` in insert_many for graceful partial failures

### Performance

- ~5-10 docs/min (depends on size, API limits)
- Batch embeddings 10-100x faster than sequential
- ~200MB base + ~50MB per large doc during processing

### Troubleshooting

- "No module named pymongo": Run `uv sync`
- "Connection refused": Check Atlas network access (whitelist IP)
- "Authentication failed": Verify MongoDB URI credentials
- "Index not found": Create in Atlas UI, not programmatically
- "Dimension mismatch": Ensure text-embedding-3-small uses 1536 dims
- "No chunks created": Missing DoclingDocument - verify _read_document() returns tuple
