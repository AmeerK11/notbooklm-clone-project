# Database Schema

This document describes the current database schema defined in `data/models.py`.

## Engine and Initialization
- ORM: SQLAlchemy 2.x
- Base class: `data.db.Base`
- Default database URL: `sqlite:///./notebooklm.db`
- Init entrypoint: `data.db.init_db()`

## Entity Relationship Overview
- `users` 1:N `oauth_accounts`
- `users` 1:N `documents`
- `documents` 1:N `chunks`
- `users` 1:N `conversations`
- `conversations` 1:N `messages`

## Tables

### `users`
Stores app users.

Columns:
- `id` INTEGER, PK
- `email` VARCHAR(255), nullable, UNIQUE, indexed
- `display_name` VARCHAR(255), nullable
- `avatar_url` VARCHAR(1024), nullable
- `is_active` BOOLEAN, NOT NULL, default `true`
- `created_at` DATETIME(timezone=True), NOT NULL, default `now()`
- `updated_at` DATETIME(timezone=True), NOT NULL, default `now()`, auto-updated on row update

Relationships:
- One-to-many with `oauth_accounts`
- One-to-many with `documents`
- One-to-many with `conversations`

Indexes and constraints:
- PK: `id`
- UNIQUE: `email`
- INDEX: `email` (implicit from `index=True`)

---

### `oauth_accounts`
OAuth provider identities linked to users (supports Hugging Face via `provider='huggingface'`).

Columns:
- `id` INTEGER, PK
- `user_id` INTEGER, NOT NULL, FK -> `users.id` ON DELETE CASCADE
- `provider` VARCHAR(50), NOT NULL, indexed
- `provider_user_id` VARCHAR(255), NOT NULL, indexed
- `username` VARCHAR(255), nullable
- `access_token` TEXT, nullable
- `refresh_token` TEXT, nullable
- `token_type` VARCHAR(50), nullable
- `scope` TEXT, nullable
- `expires_at` DATETIME(timezone=True), nullable
- `created_at` DATETIME(timezone=True), NOT NULL, default `now()`
- `updated_at` DATETIME(timezone=True), NOT NULL, default `now()`, auto-updated on row update

Relationships:
- Many-to-one with `users`

Indexes and constraints:
- PK: `id`
- UNIQUE: (`provider`, `provider_user_id`) as `uq_provider_user`
- INDEX: (`user_id`, `provider`) as `ix_oauth_user_provider`
- INDEX: `provider` (implicit)
- INDEX: `provider_user_id` (implicit)

---

### `documents`
Uploaded/ingested source documents owned by users.

Columns:
- `id` INTEGER, PK
- `user_id` INTEGER, NOT NULL, FK -> `users.id` ON DELETE CASCADE
- `title` VARCHAR(255), NOT NULL
- `source_filename` VARCHAR(1024), nullable
- `source_type` VARCHAR(50), NOT NULL, default `'upload'`
- `storage_path` VARCHAR(1024), nullable
- `summary` TEXT, nullable
- `created_at` DATETIME(timezone=True), NOT NULL, default `now()`
- `updated_at` DATETIME(timezone=True), NOT NULL, default `now()`, auto-updated on row update

Relationships:
- Many-to-one with `users`
- One-to-many with `chunks`

Indexes and constraints:
- PK: `id`
- INDEX: (`user_id`, `created_at`) as `ix_documents_user_created`

---

### `chunks`
Document chunks for retrieval and embedding linkage.

Columns:
- `id` INTEGER, PK
- `document_id` INTEGER, NOT NULL, FK -> `documents.id` ON DELETE CASCADE
- `chunk_index` INTEGER, NOT NULL
- `content` TEXT, NOT NULL
- `token_count` INTEGER, nullable
- `embedding_id` VARCHAR(255), nullable, indexed
- `created_at` DATETIME(timezone=True), NOT NULL, default `now()`

Relationships:
- Many-to-one with `documents`

Indexes and constraints:
- PK: `id`
- UNIQUE: (`document_id`, `chunk_index`) as `uq_document_chunk_index`
- INDEX: (`document_id`, `chunk_index`) as `ix_chunks_document_index`
- INDEX: `embedding_id` (implicit)

---

### `conversations`
User chat sessions.

Columns:
- `id` INTEGER, PK
- `user_id` INTEGER, NOT NULL, FK -> `users.id` ON DELETE CASCADE
- `title` VARCHAR(255), nullable
- `created_at` DATETIME(timezone=True), NOT NULL, default `now()`
- `updated_at` DATETIME(timezone=True), NOT NULL, default `now()`, auto-updated on row update

Relationships:
- Many-to-one with `users`
- One-to-many with `messages`

Indexes and constraints:
- PK: `id`
- INDEX: (`user_id`, `created_at`) as `ix_conversations_user_created`

---

### `messages`
Conversation messages, including optional citation payload.

Columns:
- `id` INTEGER, PK
- `conversation_id` INTEGER, NOT NULL, FK -> `conversations.id` ON DELETE CASCADE
- `role` VARCHAR(20), NOT NULL
- `content` TEXT, NOT NULL
- `citations` JSON, nullable
- `created_at` DATETIME(timezone=True), NOT NULL, default `now()`

Relationships:
- Many-to-one with `conversations`

Indexes and constraints:
- PK: `id`
- INDEX: (`conversation_id`, `created_at`) as `ix_messages_conversation_created`

## Notes
- Cascading deletes are enabled from parent to child via foreign keys (`ON DELETE CASCADE`).
- No soft-delete columns currently exist.
- Migration tooling (e.g., Alembic) is not yet configured; current initialization uses `Base.metadata.create_all(...)`.
