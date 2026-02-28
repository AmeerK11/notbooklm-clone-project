# ER Diagram

```mermaid
erDiagram
    users ||--o{ notebooks : owns
    notebooks ||--o{ sources : contains
    notebooks ||--o{ chat_threads : has
    chat_threads ||--o{ messages : contains
    messages ||--o{ message_citations : has
    sources ||--o{ message_citations : cited_by
    notebooks ||--o{ artifacts : generates

    users {
        int id PK
        string email UK
        string display_name
        string avatar_url
        datetime created_at
    }

    notebooks {
        int id PK
        int owner_user_id FK
        string title
        datetime created_at
        datetime updated_at
    }

    sources {
        int id PK
        int notebook_id FK
        string type
        string title
        string original_name
        string url
        string storage_path
        string status
        datetime ingested_at
    }

    chat_threads {
        int id PK
        int notebook_id FK
        string title
        datetime created_at
    }

    messages {
        int id PK
        int thread_id FK
        string role
        text content
        datetime created_at
    }

    message_citations {
        int id PK
        int message_id FK
        int source_id FK
        string chunk_ref
        text quote
        float score
    }

    artifacts {
        int id PK
        int notebook_id FK
        string type
        string title
        string status
        string file_path
        json metadata
        text content
        text error_message
        datetime created_at
        datetime generated_at
    }
```

## Notes
- User isolation is enforced through ownership on `notebooks.owner_user_id`.
- Thread, source, citation, and artifact records are notebook-scoped.
- Artifact metadata is stored in JSON (`artifacts.metadata`).
