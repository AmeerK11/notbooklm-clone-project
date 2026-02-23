# ER Diagram

```mermaid
erDiagram
    users ||--o{ oauth_accounts : has
    users ||--o{ documents : owns
    users ||--o{ conversations : owns
    documents ||--o{ chunks : contains
    conversations ||--o{ messages : contains

    users {
        int id PK
        string email UK
        string display_name
        string avatar_url
        boolean is_active
        datetime created_at
        datetime updated_at
    }

    oauth_accounts {
        int id PK
        int user_id FK
        string provider
        string provider_user_id
        string username
        text access_token
        text refresh_token
        string token_type
        text scope
        datetime expires_at
        datetime created_at
        datetime updated_at
    }

    documents {
        int id PK
        int user_id FK
        string title
        string source_filename
        string source_type
        string storage_path
        text summary
        datetime created_at
        datetime updated_at
    }

    chunks {
        int id PK
        int document_id FK
        int chunk_index
        text content
        int token_count
        string embedding_id
        datetime created_at
    }

    conversations {
        int id PK
        int user_id FK
        string title
        datetime created_at
        datetime updated_at
    }

    messages {
        int id PK
        int conversation_id FK
        string role
        text content
        json citations
        datetime created_at
    }
```

## Notes
- `oauth_accounts.provider` supports Hugging Face (`"huggingface"`).
- Composite unique constraints are documented in `DATABASE_SCHEMA.md`:
  - `uq_provider_user` on (`provider`, `provider_user_id`)
  - `uq_document_chunk_index` on (`document_id`, `chunk_index`)
