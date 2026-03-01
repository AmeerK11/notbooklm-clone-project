from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from data.models import Artifact, ChatThread, Message, MessageCitation, Notebook, Source, User


def get_or_create_user(
    db: Session,
    user_id: int = 1,
    email: str = "dev@example.com",
    display_name: str = "Dev User",
) -> User:
    user = db.get(User, user_id)
    if user:
        return user

    user = User(id=user_id, email=email, display_name=display_name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def get_or_create_user_by_email(
    db: Session,
    email: str,
    display_name: str | None = None,
    avatar_url: str | None = None,
) -> User:
    normalized_email = email.strip().lower()
    user = db.query(User).filter(User.email == normalized_email).first()
    if user:
        changed = False
        if display_name and user.display_name != display_name:
            user.display_name = display_name
            changed = True
        if avatar_url and user.avatar_url != avatar_url:
            user.avatar_url = avatar_url
            changed = True
        if changed:
            db.commit()
            db.refresh(user)
        return user

    user = User(
        email=normalized_email,
        display_name=display_name,
        avatar_url=avatar_url,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_notebook(db: Session, owner_user_id: int, title: str) -> Notebook:
    notebook = Notebook(owner_user_id=owner_user_id, title=title)
    db.add(notebook)
    db.commit()
    db.refresh(notebook)
    return notebook


def list_notebooks(db: Session, owner_user_id: int) -> list[Notebook]:
    return (
        db.query(Notebook)
        .filter(Notebook.owner_user_id == owner_user_id)
        .order_by(Notebook.created_at.desc())
        .all()
    )


def get_notebook_for_user(db: Session, notebook_id: int, owner_user_id: int) -> Notebook | None:
    return (
        db.query(Notebook)
        .filter(Notebook.id == notebook_id, Notebook.owner_user_id == owner_user_id)
        .first()
    )


def update_notebook_title(db: Session, notebook: Notebook, title: str) -> Notebook:
    notebook.title = title
    db.commit()
    db.refresh(notebook)
    return notebook


def delete_notebook(db: Session, notebook: Notebook) -> None:
    db.delete(notebook)
    db.commit()


def create_source(
    db: Session,
    notebook_id: int,
    source_type: str,
    title: str | None,
    original_name: str | None,
    url: str | None,
    storage_path: str | None,
    status: str = "pending",
) -> Source:
    source = Source(
        notebook_id=notebook_id,
        type=source_type,
        title=title,
        original_name=original_name,
        url=url,
        storage_path=storage_path,
        status=status,
    )
    db.add(source)
    db.commit()
    db.refresh(source)
    return source


def list_sources_for_notebook(db: Session, notebook_id: int) -> list[Source]:
    return (
        db.query(Source)
        .filter(Source.notebook_id == notebook_id)
        .order_by(Source.id.desc())
        .all()
    )


def update_source_status(
    db: Session,
    source_id: int,
    status: str,
    ingested_at: datetime | None = None,
) -> Source | None:
    source = db.get(Source, source_id)
    if source is None:
        return None

    source.status = status
    if ingested_at is not None:
        source.ingested_at = ingested_at
    db.commit()
    db.refresh(source)
    return source


def create_chat_thread(db: Session, notebook_id: int, title: str | None = None) -> ChatThread:
    thread = ChatThread(notebook_id=notebook_id, title=title)
    db.add(thread)
    db.commit()
    db.refresh(thread)
    return thread


def list_chat_threads(db: Session, notebook_id: int) -> list[ChatThread]:
    return (
        db.query(ChatThread)
        .filter(ChatThread.notebook_id == notebook_id)
        .order_by(ChatThread.created_at.desc())
        .all()
    )


def get_thread_for_notebook(db: Session, notebook_id: int, thread_id: int) -> ChatThread | None:
    return (
        db.query(ChatThread)
        .filter(ChatThread.id == thread_id, ChatThread.notebook_id == notebook_id)
        .first()
    )


def create_message(db: Session, thread_id: int, role: str, content: str) -> Message:
    message = Message(thread_id=thread_id, role=role, content=content)
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def list_messages_for_thread(db: Session, thread_id: int) -> list[Message]:
    return (
        db.query(Message)
        .filter(Message.thread_id == thread_id)
        .order_by(Message.created_at.asc())
        .all()
    )


def create_message_citations(
    db: Session,
    message_id: int,
    citations: list[dict[str, int | str | float | None]],
) -> list[MessageCitation]:
    rows: list[MessageCitation] = []
    for item in citations:
        row = MessageCitation(
            message_id=message_id,
            source_id=int(item["source_id"]),
            chunk_ref=item.get("chunk_ref"),  # type: ignore[arg-type]
            quote=item.get("quote"),  # type: ignore[arg-type]
            score=float(item["score"]) if item.get("score") is not None else None,
        )
        db.add(row)
        rows.append(row)
    db.commit()
    for row in rows:
        db.refresh(row)
    return rows


def list_message_citations_for_thread(
    db: Session, thread_id: int
) -> dict[int, list[dict[str, int | str | float | None]]]:
    rows = (
        db.query(MessageCitation, Source.title)
        .join(Source, Source.id == MessageCitation.source_id)
        .join(Message, Message.id == MessageCitation.message_id)
        .filter(Message.thread_id == thread_id)
        .order_by(MessageCitation.id.asc())
        .all()
    )

    citations_by_message: dict[int, list[dict[str, int | str | float | None]]] = defaultdict(list)
    for citation, source_title in rows:
        citations_by_message[int(citation.message_id)].append(
            {
                "source_id": int(citation.source_id),
                "source_title": source_title,
                "chunk_ref": citation.chunk_ref,
                "quote": citation.quote,
                "score": citation.score,
            }
        )
    return dict(citations_by_message)


def get_artifact(db: Session, artifact_id: int) -> Artifact | None:
    return db.get(Artifact, artifact_id)


def create_artifact(
    db: Session,
    notebook_id: int,
    artifact_type: str,
    title: str | None = None,
    metadata: dict | None = None,
) -> Artifact:
    artifact = Artifact(
        notebook_id=notebook_id,
        type=artifact_type,
        title=title,
        artifact_metadata=metadata or {},
        status="pending",
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


def list_artifacts(db: Session, notebook_id: int, artifact_type: str | None = None) -> list[Artifact]:
    query = db.query(Artifact).filter(Artifact.notebook_id == notebook_id)
    if artifact_type:
        query = query.filter(Artifact.type == artifact_type)
    return query.order_by(Artifact.created_at.desc()).all()


def update_artifact(
    db: Session,
    artifact_id: int,
    status: str,
    content: str | None = None,
    file_path: str | None = None,
    error_message: str | None = None,
    metadata: dict | None = None,
) -> Artifact | None:
    artifact = db.get(Artifact, artifact_id)
    if not artifact:
        return None
    
    artifact.status = status
    if content is not None:
        artifact.content = content
    if file_path is not None:
        artifact.file_path = file_path
    if error_message is not None:
        artifact.error_message = error_message
    if metadata is not None:
        merged = dict(artifact.artifact_metadata or {})
        merged.update(metadata)
        artifact.artifact_metadata = merged
    if status == "ready":
        artifact.generated_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(artifact)
    return artifact
