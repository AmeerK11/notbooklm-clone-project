# Ingestion Pipeline Specification

## Summary
- Purpose: build the ingestion pipeline that extracts text from PDFs, PPTX, TXT, and URLs; chunks text; computes embeddings; and stores embeddings + metadata in a vector database with per-user and per-notebook isolation for RAG with citations.

## Goals & Scope
- Support file and URL ingestion for per-user notebooks.
- Produce chunked, embedded documents with provenance metadata (source, page/slide, offsets).
- Provide an API/worker flow with status tracking and retries.

## High-level Architecture

- Ingestion API: Accept uploads/URLs, validate, create `source` record, enqueue ingestion job.
- Worker: Extract -> Preprocess -> Chunk -> Embed -> Upsert to vector DB -> Persist metadata.
- Storage: raw files in `data/raw/{user_id}/{notebook_id}/`, metadata in a lightweight DB (SQLite/JSON), embeddings in a vector DB (Chroma/FAISS). 

## Components

- Upload Handler: validates file type/size and writes raw file.
- Extractors: modular functions for each type (PDF, PPTX, TXT, URL).
- Preprocessing: normalize whitespace, remove boilerplate, preserve page/slide indices and char offsets.
- Chunker: token-aware sliding window with overlap; store chunk-level metadata.
- Embedder: pluggable adapter (local `sentence-transformers` or API-based embeddings).
- Vectorstore Adapter: pluggable (default Chroma). Upserts include chunk text + metadata.
- Metadata Store: tracks notebooks, sources, ingestion status, timestamps, and source enable/disable flags.
- Job Orchestrator: simple background queue with retries (Redis/RQ or asyncio-based worker).

## Extractors (recommended implementations)
- PDF: `PyMuPDF` (fitz) or `pdfminer.six`; extract per-page text and char offsets.
- PPTX: `python-pptx` extract slide text + speaker notes.
- TXT: read with encoding detection (`chardet` / `charset-normalizer`).
- URL: `requests` + `BeautifulSoup` with readability extraction; sanitize HTML.

## Chunking
- Default: 500 tokens per chunk with 50 token overlap (configurable via `CHUNK_TOKENS`, `CHUNK_OVERLAP`).
- Alternative simple default: 2000 characters with 400 char overlap for token-agnostic environments.
- Store for each chunk: `chunk_id`, `source_id`, `char_start`, `char_end`, `page`, `text_preview`.

## Embeddings
- Adapter interface supports local models (`sentence-transformers/all-MiniLM`) or API-based embeddings (OpenAI/HF).
- Configurable batch size (default 64) and rate limiting.

## Vector DB
- Default: Chroma for HF Spaces compatibility; can swap to FAISS or Weaviate later.
- Use namespacing by `user_id` + `notebook_id` or include those fields in metadata for isolation.

## Metadata Schema (suggested)
- Notebook: `{ notebook_id, user_id, name, created_at, updated_at }`
- Source: `{ source_id, notebook_id, user_id, filename, url?, status, pages, size_bytes, created_at, error? }`
- Chunk: `{ chunk_id, source_id, notebook_id, user_id, char_start, char_end, page, text_preview }`

## API Contract (ingest endpoints)
- `POST /ingest/upload` — multipart file, `user_id`, `notebook_id` → returns `{ source_id, job_id }`.
- `POST /ingest/url` — body `{ url, user_id, notebook_id }` → returns `{ source_id, job_id }`.
- `GET /ingest/status?job_id=...` → returns status and error if failed.
- `POST /ingest/enable_source` — enable/disable source for retrieval.

## Citation Strategy
- Store `source_name`, `file_path` or `url`, and `page/slide` in each chunk's metadata.
- Retrieval returns top-k chunks with metadata; present citations like `[SourceName — page 3]`.

## Job Orchestration
- Small files: synchronous ingestion (fast path).
- Large files: enqueue to background worker with retry/backoff and status updates.

## Security & Operational Considerations
- Sanitize HTML from URLs; validate file types and size limits (configurable, e.g., 100MB).
- Rate-limit embedding calls to control cost.
- Log ingestion events and basic metrics.

## File Layout (suggested)
- `ingest/`
  - `api.py` — upload + status endpoints
  - `worker.py` — ingestion worker and job runner
  - `extractors.py` — file/URL extraction functions
  - `chunker.py` — token-aware chunking utilities
  - `embedder.py` — embedding adapter
  - `vectorstore.py` — vector DB adapter
  - `metadata.py` — metadata store (SQLite/TinyDB)
- `data/raw/` and `data/meta/`

## Config / Environment Variables (examples)
- `EMBEDDING_PROVIDER`=local|openai|hf
- `CHUNK_TOKENS`=500
- `CHUNK_OVERLAP`=50
- `EMBED_BATCH_SIZE`=64
- `VECTORSTORE`=chroma|faiss|weaviate

## Acceptance Criteria
- API accepts PDF/PPTX/TXT/URL and returns `source_id` + `job_id`.
- Worker extracts text with page/slide metadata and chunks according to config.
- Embeddings stored in vector DB with metadata linking back to source and offsets.
- Retrieval returns chunk text + metadata for citation.
- Per-user and per-notebook isolation enforced.

## Testing
- Unit tests for extractors (use small sample files), chunker determinism, embedder adapter (mocked), and vectorstore upsert/retrieve (in-memory).

## CI / Deployment
- Add GitHub Actions to run tests and push to Hugging Face Space on main branch updates.

## Next Steps (immediate)
1. Scaffold `ingest/` module files and a small runner script.
2. Add unit tests and sample files for extractors and chunker.
3. Implement Chroma adapter and local `sentence-transformers` embedding support.

---
If you'd like, I can now scaffold the `ingest/` module files and tests. Also confirm preferred default embedding provider: `sentence-transformers` (local, free) or `openai/hf` (API-based).
