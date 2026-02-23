from __future__ import annotations

from contextlib import asynccontextmanager

from datetime import datetime, timezone
from pathlib import Path

from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from data import crud
from data.db import get_db, init_db
from src.ingestion.service import ingest_source, query_notebook_chunks
from utils.llm_client import LLMConfigError, generate_chat_completion
from utils.prompt_templates import build_rag_system_prompt, build_rag_user_prompt


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="NotebookLM Clone API", version="0.1.0", lifespan=lifespan)


auth_router = APIRouter(prefix="/auth", tags=["auth"])
notebooks_router = APIRouter(prefix="/notebooks", tags=["notebooks"])
sources_router = APIRouter(prefix="/sources", tags=["sources"])
threads_router = APIRouter(prefix="/threads", tags=["threads"])


class NotebookCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    owner_user_id: int | None = None


class NotebookResponse(BaseModel):
    id: int
    owner_user_id: int
    title: str


class SourceCreateRequest(BaseModel):
    owner_user_id: int | None = None
    type: str = Field(min_length=1, max_length=50)
    title: str | None = Field(default=None, max_length=255)
    original_name: str | None = Field(default=None, max_length=1024)
    url: str | None = Field(default=None, max_length=2048)
    storage_path: str | None = Field(default=None, max_length=1024)
    status: str = Field(default="pending", max_length=50)


class SourceResponse(BaseModel):
    id: int
    notebook_id: int
    type: str
    title: str | None
    original_name: str | None
    url: str | None
    storage_path: str | None
    status: str
    ingested_at: datetime | None


class ThreadCreateRequest(BaseModel):
    owner_user_id: int | None = None
    title: str | None = Field(default=None, max_length=255)


class ThreadResponse(BaseModel):
    id: int
    notebook_id: int
    title: str | None
    created_at: datetime


class MessageResponse(BaseModel):
    id: int
    thread_id: int
    role: str
    content: str
    created_at: datetime


class CitationResponse(BaseModel):
    source_title: str | None = None
    source_id: int
    chunk_ref: str | None = None
    quote: str | None = None
    score: float | None = None


class ChatRequest(BaseModel):
    owner_user_id: int | None = None
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=12)


class ChatResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse
    citations: list[CitationResponse]


MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARS_PER_MESSAGE = 1000


def _build_conversation_history(
    thread_messages: list, max_messages: int = MAX_HISTORY_MESSAGES
) -> list[str]:
    history_slice = thread_messages[-max_messages:] if len(thread_messages) > max_messages else thread_messages
    rows: list[str] = []
    for msg in history_slice:
        role = str(msg.role).strip().lower()
        content = str(msg.content or "").strip()
        if not content:
            continue
        if len(content) > MAX_HISTORY_CHARS_PER_MESSAGE:
            content = content[:MAX_HISTORY_CHARS_PER_MESSAGE] + "..."
        rows.append(f"{role}: {content}")
    return rows


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/db", tags=["system"])
def health_db(db: Session = Depends(get_db)) -> dict[str, str]:
    db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "connected"}


@auth_router.get("/status")
def auth_status() -> dict[str, str]:
    return {"message": "Auth routes not implemented yet."}


@notebooks_router.post("", response_model=NotebookResponse)
def create_notebook(payload: NotebookCreateRequest, db: Session = Depends(get_db)) -> NotebookResponse:
    # Temporary behavior until OAuth is wired: default everything to user_id=1.
    owner_user_id = payload.owner_user_id or 1
    crud.get_or_create_user(db=db, user_id=owner_user_id)
    notebook = crud.create_notebook(db=db, owner_user_id=owner_user_id, title=payload.title)
    return NotebookResponse(id=notebook.id, owner_user_id=notebook.owner_user_id, title=notebook.title)


@notebooks_router.get("", response_model=list[NotebookResponse])
def get_notebooks(owner_user_id: int = 1, db: Session = Depends(get_db)) -> list[NotebookResponse]:
    notebooks = crud.list_notebooks(db=db, owner_user_id=owner_user_id)
    return [
        NotebookResponse(id=n.id, owner_user_id=n.owner_user_id, title=n.title) for n in notebooks
    ]


@notebooks_router.post("/{notebook_id}/sources", response_model=SourceResponse)
def create_source_for_notebook(
    notebook_id: int, payload: SourceCreateRequest, db: Session = Depends(get_db)
) -> SourceResponse:
    owner_user_id = payload.owner_user_id or 1
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=owner_user_id
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    source = crud.create_source(
        db=db,
        notebook_id=notebook_id,
        source_type=payload.type,
        title=payload.title,
        original_name=payload.original_name,
        url=payload.url,
        storage_path=payload.storage_path,
        status=payload.status,
    )
    return SourceResponse(
        id=source.id,
        notebook_id=source.notebook_id,
        type=source.type,
        title=source.title,
        original_name=source.original_name,
        url=source.url,
        storage_path=source.storage_path,
        status=source.status,
        ingested_at=source.ingested_at,
    )


@notebooks_router.post("/{notebook_id}/sources/upload", response_model=SourceResponse)
async def upload_source_for_notebook(
    notebook_id: int,
    owner_user_id: int = Form(1),
    title: str | None = Form(None),
    status: str = Form("pending"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> SourceResponse:
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=owner_user_id
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    upload_dir = Path("uploads") / f"notebook_{notebook_id}"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / file.filename
    content = await file.read()
    destination.write_bytes(content)

    source = crud.create_source(
        db=db,
        notebook_id=notebook_id,
        source_type="file",
        title=title or file.filename,
        original_name=file.filename,
        url=None,
        storage_path=str(destination),
        status=status,
    )

    crud.update_source_status(db=db, source_id=source.id, status="processing")
    try:
        ingested_chunk_count = await run_in_threadpool(
            ingest_source, source=source, owner_user_id=owner_user_id
        )
        final_status = "ready" if ingested_chunk_count > 0 else "failed"
        source = crud.update_source_status(
            db=db,
            source_id=source.id,
            status=final_status,
            ingested_at=datetime.now(timezone.utc) if final_status == "ready" else None,
        ) or source
    except Exception as exc:
        crud.update_source_status(db=db, source_id=source.id, status="failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return SourceResponse(
        id=source.id,
        notebook_id=source.notebook_id,
        type=source.type,
        title=source.title,
        original_name=source.original_name,
        url=source.url,
        storage_path=source.storage_path,
        status=source.status,
        ingested_at=source.ingested_at,
    )


@notebooks_router.get("/{notebook_id}/sources", response_model=list[SourceResponse])
def list_sources_for_notebook(
    notebook_id: int, owner_user_id: int = 1, db: Session = Depends(get_db)
) -> list[SourceResponse]:
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=owner_user_id
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    sources = crud.list_sources_for_notebook(db=db, notebook_id=notebook_id)
    return [
        SourceResponse(
            id=s.id,
            notebook_id=s.notebook_id,
            type=s.type,
            title=s.title,
            original_name=s.original_name,
            url=s.url,
            storage_path=s.storage_path,
            status=s.status,
            ingested_at=s.ingested_at,
        )
        for s in sources
    ]


@sources_router.get("")
def list_sources_placeholder() -> dict[str, str]:
    return {"message": "Use /notebooks/{notebook_id}/sources endpoints."}


@notebooks_router.post("/{notebook_id}/threads", response_model=ThreadResponse)
def create_thread_for_notebook(
    notebook_id: int, payload: ThreadCreateRequest, db: Session = Depends(get_db)
) -> ThreadResponse:
    owner_user_id = payload.owner_user_id or 1
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=owner_user_id
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    thread = crud.create_chat_thread(db=db, notebook_id=notebook_id, title=payload.title)
    return ThreadResponse(
        id=thread.id, notebook_id=thread.notebook_id, title=thread.title, created_at=thread.created_at
    )


@notebooks_router.get("/{notebook_id}/threads", response_model=list[ThreadResponse])
def list_threads_for_notebook(
    notebook_id: int, owner_user_id: int = 1, db: Session = Depends(get_db)
) -> list[ThreadResponse]:
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=owner_user_id
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    threads = crud.list_chat_threads(db=db, notebook_id=notebook_id)
    return [
        ThreadResponse(id=t.id, notebook_id=t.notebook_id, title=t.title, created_at=t.created_at)
        for t in threads
    ]


@threads_router.get("/{thread_id}/messages", response_model=list[MessageResponse])
def list_messages_for_thread(
    thread_id: int, notebook_id: int, owner_user_id: int = 1, db: Session = Depends(get_db)
) -> list[MessageResponse]:
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=owner_user_id
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    thread = crud.get_thread_for_notebook(db=db, notebook_id=notebook_id, thread_id=thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found for this notebook.")

    messages = crud.list_messages_for_thread(db=db, thread_id=thread_id)
    return [
        MessageResponse(
            id=m.id, thread_id=m.thread_id, role=m.role, content=m.content, created_at=m.created_at
        )
        for m in messages
    ]


@threads_router.post("/{thread_id}/chat", response_model=ChatResponse)
def chat_on_thread(
    thread_id: int, payload: ChatRequest, notebook_id: int, db: Session = Depends(get_db)
) -> ChatResponse:
    owner_user_id = payload.owner_user_id or 1
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=owner_user_id
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    thread = crud.get_thread_for_notebook(db=db, notebook_id=notebook_id, thread_id=thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found for this notebook.")

    prior_messages = crud.list_messages_for_thread(db=db, thread_id=thread_id)
    user_message = crud.create_message(db=db, thread_id=thread_id, role="user", content=payload.question)

    retrieval_rows = query_notebook_chunks(
        owner_user_id=owner_user_id,
        notebook_id=notebook_id,
        query_text=payload.question,
        top_k=payload.top_k,
    )

    context_blocks: list[str] = []
    citations: list[CitationResponse] = []
    citation_rows: list[dict[str, int | str | float | None]] = []
    for row in retrieval_rows:
        doc = row.get("document", "")
        meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        score = row.get("score")
        try:
            source_id = int(meta.get("source_id", 0))
        except (TypeError, ValueError):
            source_id = 0
        chunk_index = meta.get("chunk_index")
        source_title = (
            str(meta.get("source_title")) if isinstance(meta, dict) and meta.get("source_title") else None
        )
        chunk_ref = f"source_{source_id}_chunk_{chunk_index}" if source_id and chunk_index is not None else None
        context_blocks.append(
            f"[source_title={source_title or 'Unknown'}, source_id={source_id}, chunk_index={chunk_index}]\n{doc}"
        )
        citation = CitationResponse(
            source_title=source_title,
            source_id=source_id,
            chunk_ref=chunk_ref,
            quote=(doc[:300] if isinstance(doc, str) else None),
            score=(float(score) if score is not None else None),
        )
        citations.append(citation)
        if source_id:
            citation_rows.append(
                {
                    "source_id": source_id,
                    "chunk_ref": chunk_ref,
                    "quote": citation.quote,
                    "score": citation.score,
                }
            )

    if context_blocks:
        system_prompt = build_rag_system_prompt()
        conversation_history = _build_conversation_history(prior_messages)
        user_prompt = build_rag_user_prompt(
            question=payload.question,
            context_blocks=context_blocks,
            conversation_history=conversation_history,
        )
        try:
            answer = generate_chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)
        except LLMConfigError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {exc}") from exc
    else:
        answer = "I do not have enough indexed context to answer this question yet."

    assistant_message = crud.create_message(
        db=db, thread_id=thread_id, role="assistant", content=answer
    )
    if citation_rows:
        crud.create_message_citations(db=db, message_id=assistant_message.id, citations=citation_rows)

    return ChatResponse(
        user_message=MessageResponse(
            id=user_message.id,
            thread_id=user_message.thread_id,
            role=user_message.role,
            content=user_message.content,
            created_at=user_message.created_at,
        ),
        assistant_message=MessageResponse(
            id=assistant_message.id,
            thread_id=assistant_message.thread_id,
            role=assistant_message.role,
            content=assistant_message.content,
            created_at=assistant_message.created_at,
        ),
        citations=citations,
    )


@threads_router.get("")
def list_threads_placeholder() -> dict[str, str]:
    return {"message": "Use /notebooks/{notebook_id}/threads endpoints."}


app.include_router(auth_router)
app.include_router(notebooks_router)
app.include_router(sources_router)
app.include_router(threads_router)
