# Database Schema

This document reflects the active SQLAlchemy models in `data/models.py`.

## Engine and Initialization
- ORM: SQLAlchemy 2.x
- Base class: `data.db.Base`
- Default DB: `sqlite:///./notebooklm.db`
- Initialization: `data.db.init_db()`

## Relationship Overview
- `users` 1:N `notebooks`
- `notebooks` 1:N `sources`
- `notebooks` 1:N `chat_threads`
- `chat_threads` 1:N `messages`
- `messages` 1:N `message_citations`
- `sources` 1:N `message_citations`
- `notebooks` 1:N `artifacts`

## Tables

### `users`
Columns:
- `id` INTEGER PK
- `email` VARCHAR(255) NOT NULL UNIQUE INDEX
- `display_name` VARCHAR(255) NULL
- `avatar_url` VARCHAR(1024) NULL
- `created_at` DATETIME(timezone=True) NOT NULL

### `notebooks`
Columns:
- `id` INTEGER PK
- `owner_user_id` INTEGER NOT NULL FK -> `users.id` ON DELETE CASCADE INDEX
- `title` VARCHAR(255) NOT NULL
- `created_at` DATETIME(timezone=True) NOT NULL
- `updated_at` DATETIME(timezone=True) NOT NULL

### `sources`
Columns:
- `id` INTEGER PK
- `notebook_id` INTEGER NOT NULL FK -> `notebooks.id` ON DELETE CASCADE INDEX
- `type` VARCHAR(50) NOT NULL
- `title` VARCHAR(255) NULL
- `original_name` VARCHAR(1024) NULL
- `url` VARCHAR(2048) NULL
- `storage_path` VARCHAR(1024) NULL
- `status` VARCHAR(50) NOT NULL
- `ingested_at` DATETIME(timezone=True) NULL

### `chat_threads`
Columns:
- `id` INTEGER PK
- `notebook_id` INTEGER NOT NULL FK -> `notebooks.id` ON DELETE CASCADE INDEX
- `title` VARCHAR(255) NULL
- `created_at` DATETIME(timezone=True) NOT NULL

### `messages`
Columns:
- `id` INTEGER PK
- `thread_id` INTEGER NOT NULL FK -> `chat_threads.id` ON DELETE CASCADE INDEX
- `role` VARCHAR(20) NOT NULL
- `content` TEXT NOT NULL
- `created_at` DATETIME(timezone=True) NOT NULL

### `message_citations`
Columns:
- `id` INTEGER PK
- `message_id` INTEGER NOT NULL FK -> `messages.id` ON DELETE CASCADE INDEX
- `source_id` INTEGER NOT NULL FK -> `sources.id` ON DELETE CASCADE INDEX
- `chunk_ref` VARCHAR(255) NULL
- `quote` TEXT NULL
- `score` FLOAT NULL

### `artifacts`
Columns:
- `id` INTEGER PK
- `notebook_id` INTEGER NOT NULL FK -> `notebooks.id` ON DELETE CASCADE INDEX
- `type` VARCHAR(50) NOT NULL
- `title` VARCHAR(255) NULL
- `status` VARCHAR(50) NOT NULL
- `file_path` VARCHAR(1024) NULL
- `metadata` JSON NULL (mapped as `artifact_metadata`)
- `content` TEXT NULL
- `error_message` TEXT NULL
- `created_at` DATETIME(timezone=True) NOT NULL
- `generated_at` DATETIME(timezone=True) NULL

## Notes
- Ownership and isolation are anchored by `notebooks.owner_user_id`.
- Child records are deleted via `ON DELETE CASCADE`.
- Schema creation is currently handled with `Base.metadata.create_all(...)` (no Alembic yet).
