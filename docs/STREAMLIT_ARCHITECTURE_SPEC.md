# NotebookLM Clone Streamlit Architecture Spec

## 1. Scope
This spec defines the target MVP for a Streamlit-based NotebookLM clone with:
- source ingestion (`.pdf`, `.pptx`, `.txt`, URL)
- retrieval-augmented chat with citations
- artifact generation (report, quiz, podcast transcript/audio)
- strict per-user and per-notebook data isolation
- CI/CD deployment from GitHub to Hugging Face Spaces

This document aligns your course requirements, the initial plan PDF, and the current repository implementation.

## 2. Current Baseline (Repo Audit)
Implemented now:
- FastAPI backend with notebook/source/thread/chat endpoints in `app.py`
- Streamlit frontend in `frontend/app.py`
- ingestion pipeline in `src/ingestion/` (extract, chunk, embed, Chroma upsert/query)
- SQLite schema and CRUD in `data/models.py` and `data/crud.py`
- artifact endpoints for report, quiz, and podcast in `app.py`
- HF OAuth + session bridge integration in `auth/oauth.py` and `auth/session.py`
- notebook create/rename/delete and notebook-scoped source/thread/artifact routes
- URL ingestion safety controls (scheme allowlist, DNS/IP checks, redirect/body limits)
- URL source auto-ingestion (`processing -> ready/failed`) and file upload ingestion
- per-user authorization checks via `require_current_user`
- Streamlit artifact panel with preview/download/playback controls
- GitHub Actions workflow for deploy to Hugging Face Space

Remaining MVP gaps / hardening:
- citation history display should persist clearly when reloading existing threads
- operational docs/runbook should be updated with final artifact output formats and auth/deploy setup

## 3. Target Architecture (Streamlit + FastAPI)

### 3.1 Frontend (Streamlit)
- `frontend/app.py` remains the primary UI.
- Pages/sections:
  - Auth/session status
  - Notebook manager (create, rename, delete, switch)
  - Source ingestion (upload + URL)
  - Chat panel with citations
  - Artifact panel (generate/list/download/playback)
- Session state stores selected notebook/thread and user identity from OAuth.

### 3.2 Backend (FastAPI)
- Keep `app.py` routers for API boundaries:
  - `/auth/*`
  - `/notebooks/*`
  - `/threads/*`
  - `/sources/*`
  - `/notebooks/{id}/artifacts/*`
- Service layer responsibilities:
  - ingestion orchestration (`src/ingestion/service.py`)
  - RAG retrieval + prompt construction (`query_notebook_chunks`, prompt templates)
  - artifact generation (`src/artifacts/*`)
- Authorization rule: every notebook/thread/source/artifact operation must verify ownership against authenticated user.

### 3.3 Storage
Primary metadata store:
- SQLite (`users`, `notebooks`, `sources`, `chat_threads`, `messages`, `message_citations`, `artifacts`)

Vector store:
- ChromaDB collections with notebook scoping metadata (`user_id`, `notebook_id`, `source_id`, chunk refs)

File/object storage layout (MVP local/HF `/data`):
```
/data/users/<username>/notebooks/<notebook_uuid>/
  files_raw/
  files_extracted/
  chroma/
  chat/messages.jsonl
  artifacts/reports/
  artifacts/quizzes/
  artifacts/podcasts/
```

## 4. Identity, Auth, and Isolation Plan

### 4.1 Authentication
- Integrate Hugging Face OAuth for user login.
- Map provider identity (`hf_sub` or stable username) to internal `users` row.
- Store session in secure cookie/server session.

### 4.2 Authorization
- Replace free-form `owner_user_id` from UI with server-derived user ID from session.
- Add shared helper (dependency/middleware) to resolve `current_user`.
- Enforce ownership checks in every read/write endpoint.

### 4.3 Isolation invariants
- DB queries always include ownership constraints.
- Vector queries include `user_id` and `notebook_id` metadata filters.
- File paths are derived from trusted IDs only (never direct user path input).

## 5. Functional Requirements and API Plan

### 5.1 Notebook lifecycle
Required:
- create notebook
- list notebooks for current user
- rename notebook
- delete notebook

Backend additions:
- `PATCH /notebooks/{notebook_id}`
- `DELETE /notebooks/{notebook_id}`

### 5.2 Source ingestion
Required:
- upload `.pdf/.pptx/.txt` files
- ingest URL sources with safe fetch rules
- extract, chunk, embed, store, mark ready/failed

Backend additions:
- URL validator + SSRF guardrail module (block private IP ranges, non-http(s), large responses)

### 5.3 RAG chat with citations
Required:
- retrieve top-k notebook chunks
- generate answer grounded in retrieved context
- return citation metadata and persist messages + citations

Current state:
- mostly implemented in `POST /threads/{thread_id}/chat`

Hardening needed:
- stronger citation formatting in responses
- conversation token budgeting and truncation policy

### 5.4 Artifact generation
Required outputs:
- report (`.md`)
- quiz (`.md` + answer key)
- podcast transcript (`.md`) + audio (`.mp3`)

Current state:
- all three artifact endpoints exist and are wired in Streamlit
- report output is persisted as Markdown (`.md`)
- quiz output is persisted as Markdown (`.md`) including answer key
- podcast persists transcript Markdown (`.md`) and audio (`.mp3`)

Backend additions:
- standard artifact serialization + saved output files under artifact subfolders

### 5.5 UI requirements
Required frontend features:
- notebook manager with switching
- source upload + URL ingest
- chat with visible citations
- artifact generate buttons (report/quiz/podcast)
- artifact list with download links
- podcast playback component
- explicit error/retry states

## 6. CI/CD Requirements (GitHub -> HF Space)
- Trigger on push to `main`.
- Sync repository to HF Space via token auth.
- Use GitHub Secrets:
  - `HF_TOKEN`
  - `HF_SPACE_REPO` (example: `username/space-name`)
  - optional: `HF_SPACE_BRANCH` (default `main`)
- Optional pre-deploy check: run tests before sync.

## 7. Milestone Plan

### Milestone 1: Auth + Isolation foundation
- Implement HF OAuth and session plumbing.
- Remove manual `owner_user_id` UI field.
- Add authorization dependency and enforce route coverage.

Exit criteria:
- no endpoint accepts cross-user notebook/thread access.

### Milestone 2: Notebook + Ingestion completeness
- add notebook rename/delete APIs and UI actions.
- add SSRF-safe URL ingestion policy.
- improve ingestion status feedback in UI.

Exit criteria:
- complete notebook lifecycle and safe ingestion of all required source types.

### Milestone 3: RAG + Artifacts
- improve chat citation UX and persistence views.
- add report artifact generation + storage.
- finalize artifact browser/download/audio playback in Streamlit.

Exit criteria:
- all three artifact types are generated, listed, and downloadable/playable.

### Milestone 4: Deployment hardening
- enable GitHub Actions HF deploy.
- add smoke test steps and env validation.
- document operational runbook.

Exit criteria:
- push to `main` updates HF Space automatically.

## 8. Risk Controls
- Cost control: cap tokens, default economical model, per-request limits.
- Ephemeral storage: keep extracted text/chunks to rebuild vectors.
- Prompt injection: treat source text as untrusted and constrain system prompts.
- URL ingestion abuse: protocol allowlist, IP range blocklist, timeout/size caps.
- Dependency risk: pin versions, scan vulnerabilities in CI periodically.

## 9. Build Order (Recommended Next 10 Tasks)
1. implement `current_user` auth dependency
2. wire HF OAuth callbacks
3. replace UI `owner_user_id` with authenticated identity
4. add notebook rename API + UI
5. add notebook delete API + UI confirmation
6. add report artifact generator + endpoint
7. add artifact list/download/playback panel in Streamlit
8. add URL safety validator module for ingestion
9. add integration tests for cross-user isolation
10. enforce CI deploy workflow and add README deployment setup
