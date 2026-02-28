# NotebookLM Clone Design Brief

## 1. System Overview
This system is a full-stack NotebookLM-style application that supports:
- source ingestion (`.pdf`, `.pptx`, `.txt`, web URL)
- retrieval-augmented chat with citations
- artifact generation (report, quiz, podcast transcript + audio)
- strict per-user data isolation with multiple notebooks per user

The stack is optimized for Hugging Face Spaces deployment:
- frontend: Streamlit (`frontend/app.py`)
- backend API: FastAPI (`app.py`)
- metadata store: SQLite via SQLAlchemy (`data/models.py`, `data/crud.py`)
- vector store: ChromaDB per user+notebook (`src/ingestion/vectorstore.py`)
- ingestion/artifact services: `src/ingestion/*`, `src/artifacts/*`

## 2. Architecture Diagram
```mermaid
flowchart TD
    A[Streamlit Frontend] --> B[FastAPI Backend]
    B --> C[Auth Layer<br/>HF OAuth / Dev Auth]
    B --> D[Notebook & Source APIs]
    B --> E[Thread & Chat APIs]
    B --> F[Artifact APIs]

    D --> G[Ingestion Service]
    G --> H[Extractors<br/>PDF/PPTX/TXT/URL]
    G --> I[Chunker]
    G --> J[Embedding Adapter]
    G --> K[ChromaDB]

    E --> K
    E --> L[LLM Client]
    E --> M[Message + Citation Tables]

    F --> L
    F --> N[TTS Adapter<br/>Edge/OpenAI/ElevenLabs]
    F --> O[Artifacts on Disk]

    B --> P[(SQLite DB)]
    B --> Q[/data + uploads Storage]
```

## 3. Component Responsibilities
- `frontend/app.py`
  - authentication-aware UI
  - notebook switching
  - source upload/URL ingestion
  - chat interface + citation display
  - artifact generation, preview, and downloads
- `app.py`
  - route orchestration and auth enforcement
  - notebook/source/thread/artifact lifecycle endpoints
  - chat orchestration with retrieval + prompting
  - background podcast generation
- `auth/oauth.py`, `auth/session.py`
  - HF OAuth code exchange
  - secure session bridging to Streamlit
  - current-user resolution
- `src/ingestion/*`
  - extraction, chunking, embedding, vector upsert/query
- `src/artifacts/*`
  - report/quiz/podcast generation and storage
  - pluggable TTS providers (`edge`, `openai`, `elevenlabs`)
- `data/models.py`, `data/crud.py`
  - relational schema and ownership-scoped queries

## 4. Data Model and Storage Strategy
Relational entities:
- `users`
- `notebooks` (`owner_user_id` foreign key)
- `sources` (per notebook)
- `chat_threads` and `messages`
- `message_citations` (assistant message -> source references)
- `artifacts` (status, metadata, content, file path)

Filesystem layout:
```text
<STORAGE_BASE_DIR>/users/<user_id>/notebooks/<notebook_id>/
  files_raw/
  files_extracted/
  chroma/
  artifacts/reports/
  artifacts/quizzes/
  artifacts/podcasts/
uploads/notebook_<notebook_id>/
```

Design rationale:
- SQLite keeps operational complexity low for MVP.
- Chroma per notebook enables practical RAG retrieval with low infra overhead.
- Disk layout mirrors ownership boundaries for simple cleanup and auditability.

## 5. End-to-End Flow
### 5.1 Ingestion
1. User uploads file or submits URL from Streamlit.
2. Backend verifies notebook ownership and validates URL safety (if URL).
3. Source record is created with `processing` status.
4. Ingestion service extracts text, chunks, embeds, and upserts into Chroma.
5. Source status transitions to `ready` or `failed`.

### 5.2 Retrieval + Chat
1. User sends a message in a notebook thread.
2. Backend checks notebook/thread ownership.
3. Query embedding is computed and top-k chunks are retrieved from notebook Chroma.
4. Prompt is assembled with conversation history and retrieved context.
5. LLM generates an answer.
6. Assistant message and structured citations are persisted.
7. UI shows answer and citations; citations remain available on subsequent reloads.

## 6. Security Plan
Authentication and identity:
- `AUTH_MODE=hf_oauth` for production deployments.
- Session-based current-user identity with signed bridge tokens.

User isolation:
- all notebook/thread/source/artifact endpoints verify ownership (`owner_user_id`)
- retrieval path binds queries to current user and notebook

Path/data protection:
- upload filenames are sanitized and constrained to notebook upload roots
- deletion is bounded to expected storage roots to prevent unsafe recursive deletes
- URL ingestion blocks local/private network targets (SSRF reduction)

Operational controls:
- environment-based secrets (`APP_SESSION_SECRET`, API keys)
- CI test gate before deploy

## 7. Milestone Plan
### MVP (Milestone 1)
- auth + sessions
- notebook CRUD + isolation checks
- ingestion for PDF/PPTX/TXT/URL
- notebook-scoped RAG chat with citations

### Milestone 2
- artifact generation endpoints (report/quiz/podcast)
- transcript/audio persistence and frontend playback/download
- improved chat UX and citation persistence in history

### Milestone 3 (Extensions)
- compare retrieval techniques (baseline semantic vs hybrid/rerank)
- latency/quality benchmarking and report
- stronger observability and error analytics

## 8. Key Risks and Mitigations
- LLM/API cost volatility
  - mitigate with model selection defaults, request limits, caching opportunities
- HF `/data` ephemerality on free tier
  - document tradeoff; optional HF dataset persistence extension
- retrieval quality drift across document types
  - tune chunking and top-k; evaluate reranking/hybrid methods
- URL ingestion abuse
  - strict scheme/host/IP/redirect/content-size checks
- dependency/runtime mismatch
  - CI tests and pinned dependency strategy where practical

## 9. Specifications and References in Repo
- ingestion spec: `docs/INGESTION_SPEC.md`
- architecture spec: `docs/STREAMLIT_ARCHITECTURE_SPEC.md`
- integration notes: `INTEGRATION.md`
- schema docs: `ER_DIAGRAM.md`, `DATABASE_SCHEMA.md`

This brief is intended for export to PDF as the 2-4 page design deliverable.
