from __future__ import annotations

from contextlib import asynccontextmanager
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from auth.oauth import HFOAuthError, build_hf_authorize_url, exchange_code_for_hf_user, generate_oauth_state
from auth.session import (
    AUTH_MODE_DEV,
    AUTH_MODE_HF,
    AuthBridgeTokenError,
    CurrentUser,
    clear_session_user,
    configure_session_middleware,
    decode_auth_bridge_token,
    generate_auth_bridge_token,
    get_auth_mode,
    get_session_user,
    require_current_user,
    set_session_user,
)
from data import crud
from data.db import get_db, init_db, SessionLocal
from src.ingestion.extractors import URLValidationError, validate_ingestion_url
from src.ingestion.service import ingest_source, query_notebook_chunks
from src.artifacts.report_generator import ReportGenerator
from src.artifacts.quiz_generator import QuizGenerator
from src.artifacts.podcast_generator import PodcastGenerator
from utils.llm_client import LLMConfigError, generate_chat_completion
from utils.prompt_templates import build_rag_system_prompt, build_rag_user_prompt


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="NotebookLM Clone API", version="0.1.0", lifespan=lifespan)
configure_session_middleware(app)


auth_router = APIRouter(prefix="/auth", tags=["auth"])
notebooks_router = APIRouter(prefix="/notebooks", tags=["notebooks"])
sources_router = APIRouter(prefix="/sources", tags=["sources"])
threads_router = APIRouter(prefix="/threads", tags=["threads"])


class NotebookCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class NotebookUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class NotebookResponse(BaseModel):
    id: int
    owner_user_id: int
    title: str


class SourceCreateRequest(BaseModel):
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
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=12)


class ChatResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse
    citations: list[CitationResponse]


class QuizGenerateRequest(BaseModel):
    num_questions: int = Field(default=5, ge=1, le=20)
    difficulty: str = Field(default="medium")
    topic_focus: str | None = None
    title: str | None = None


class ReportGenerateRequest(BaseModel):
    detail_level: str = Field(default="medium")
    topic_focus: str | None = None
    title: str | None = None


class PodcastGenerateRequest(BaseModel):
    duration: str = Field(default="5min")
    topic_focus: str | None = None
    title: str | None = None


class DevLoginRequest(BaseModel):
    email: str | None = None
    display_name: str | None = None


class SessionUserResponse(BaseModel):
    id: int
    email: str
    display_name: str | None = None
    avatar_url: str | None = None


class AuthStatusResponse(BaseModel):
    mode: str
    authenticated: bool
    user: SessionUserResponse | None = None
    login_url: str | None = None


class AuthBridgeExchangeRequest(BaseModel):
    token: str = Field(min_length=1)


class NotebookDeleteResponse(BaseModel):
    status: str
    notebook_id: int


class ArtifactResponse(BaseModel):
    id: int
    notebook_id: int
    type: str
    title: str | None
    status: str
    content: str | None
    file_path: str | None
    metadata: dict | None
    error_message: str | None
    created_at: datetime
    generated_at: datetime | None


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


def _auth_callback_url(request: Request) -> str:
    configured = os.getenv("HF_OAUTH_REDIRECT_URI", "").strip()
    if configured:
        return configured
    base = str(request.base_url).rstrip("/")
    return f"{base}/auth/callback"


def _auth_status_payload(request: Request) -> AuthStatusResponse:
    mode = get_auth_mode()
    session_user = get_session_user(request)
    user_payload = (
        SessionUserResponse(
            id=session_user.id,
            email=session_user.email,
            display_name=session_user.display_name,
            avatar_url=session_user.avatar_url,
        )
        if session_user
        else None
    )
    return AuthStatusResponse(
        mode=mode,
        authenticated=(session_user is not None),
        user=user_payload,
        login_url="/auth/login" if mode == AUTH_MODE_HF else None,
    )


def _append_query_param(url: str, key: str, value: str) -> str:
    split = urlsplit(url)
    query_items = dict(parse_qsl(split.query, keep_blank_values=True))
    query_items[key] = value
    updated_query = urlencode(query_items)
    return urlunsplit((split.scheme, split.netloc, split.path, updated_query, split.fragment))


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {
        "message": "NotebookLM Clone API",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health/db", tags=["system"])
def health_db(db: Session = Depends(get_db)) -> dict[str, str]:
    db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "connected"}


@auth_router.get("/status", response_model=AuthStatusResponse)
def auth_status(request: Request) -> AuthStatusResponse:
    return _auth_status_payload(request)


@auth_router.post("/dev-login", response_model=AuthStatusResponse)
def auth_dev_login(
    payload: DevLoginRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> AuthStatusResponse:
    if get_auth_mode() != AUTH_MODE_DEV:
        raise HTTPException(status_code=400, detail="Dev login is disabled for this deployment.")

    default_email = os.getenv("AUTH_DEV_EMAIL", "dev@example.com")
    default_name = os.getenv("AUTH_DEV_DISPLAY_NAME", "Dev User")
    email = (payload.email or default_email).strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="A valid email is required.")
    display_name = (payload.display_name or default_name).strip() or None

    user = crud.get_or_create_user_by_email(
        db=db,
        email=email,
        display_name=display_name,
    )
    set_session_user(
        request,
        CurrentUser(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            avatar_url=user.avatar_url,
        ),
    )
    return _auth_status_payload(request)


@auth_router.post("/logout", response_model=AuthStatusResponse)
def auth_logout(request: Request) -> AuthStatusResponse:
    clear_session_user(request)
    return _auth_status_payload(request)


@auth_router.post("/bridge/exchange", response_model=AuthStatusResponse)
def auth_bridge_exchange(
    payload: AuthBridgeExchangeRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> AuthStatusResponse:
    if get_auth_mode() != AUTH_MODE_HF:
        raise HTTPException(status_code=400, detail="Auth bridge is only available in hf_oauth mode.")

    try:
        bridged_identity = decode_auth_bridge_token(payload.token)
    except AuthBridgeTokenError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user = crud.get_or_create_user_by_email(
        db=db,
        email=bridged_identity.email,
        display_name=bridged_identity.display_name,
        avatar_url=bridged_identity.avatar_url,
    )
    set_session_user(
        request,
        CurrentUser(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            avatar_url=user.avatar_url,
        ),
    )
    return _auth_status_payload(request)


@auth_router.get("/login")
def auth_login(request: Request) -> RedirectResponse:
    if get_auth_mode() != AUTH_MODE_HF:
        raise HTTPException(status_code=400, detail="HF OAuth is not enabled.")
    state = generate_oauth_state()
    request.session["oauth_state"] = state
    return RedirectResponse(
        url=build_hf_authorize_url(redirect_uri=_auth_callback_url(request), state=state),
        status_code=status.HTTP_302_FOUND,
    )


@auth_router.get("/callback")
async def auth_callback(request: Request, db: Session = Depends(get_db)) -> RedirectResponse:
    if get_auth_mode() != AUTH_MODE_HF:
        raise HTTPException(status_code=400, detail="HF OAuth is not enabled.")

    expected_state = request.session.get("oauth_state")
    state = request.query_params.get("state")
    code = request.query_params.get("code")
    if not expected_state or not state or state != expected_state:
        raise HTTPException(status_code=400, detail="Invalid OAuth state.")
    if not code:
        raise HTTPException(status_code=400, detail="Missing OAuth code.")

    try:
        identity = await exchange_code_for_hf_user(code=code, redirect_uri=_auth_callback_url(request))
    except HFOAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user = crud.get_or_create_user_by_email(
        db=db,
        email=identity["email"],
        display_name=identity.get("display_name"),
        avatar_url=identity.get("avatar_url"),
    )
    set_session_user(
        request,
        CurrentUser(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            avatar_url=user.avatar_url,
        ),
    )
    request.session.pop("oauth_state", None)
    bridge_token = generate_auth_bridge_token(
        CurrentUser(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            avatar_url=user.avatar_url,
        )
    )
    redirect_url = _append_query_param(
        os.getenv("AUTH_SUCCESS_REDIRECT_URL", "/"),
        "auth_bridge",
        bridge_token,
    )
    return RedirectResponse(
        url=redirect_url,
        status_code=status.HTTP_302_FOUND,
    )


@notebooks_router.post("", response_model=NotebookResponse)
def create_notebook(
    payload: NotebookCreateRequest,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> NotebookResponse:
    notebook = crud.create_notebook(db=db, owner_user_id=current_user.id, title=payload.title)
    return NotebookResponse(id=notebook.id, owner_user_id=notebook.owner_user_id, title=notebook.title)


@notebooks_router.get("", response_model=list[NotebookResponse])
def get_notebooks(
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> list[NotebookResponse]:
    notebooks = crud.list_notebooks(db=db, owner_user_id=current_user.id)
    return [
        NotebookResponse(id=n.id, owner_user_id=n.owner_user_id, title=n.title) for n in notebooks
    ]


@notebooks_router.patch("/{notebook_id}", response_model=NotebookResponse)
def rename_notebook(
    notebook_id: int,
    payload: NotebookUpdateRequest,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> NotebookResponse:
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=current_user.id
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    updated = crud.update_notebook_title(db=db, notebook=notebook, title=payload.title.strip())
    return NotebookResponse(
        id=updated.id,
        owner_user_id=updated.owner_user_id,
        title=updated.title,
    )


@notebooks_router.delete("/{notebook_id}", response_model=NotebookDeleteResponse)
def delete_notebook(
    notebook_id: int,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> NotebookDeleteResponse:
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=current_user.id
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    crud.delete_notebook(db=db, notebook=notebook)
    return NotebookDeleteResponse(status="deleted", notebook_id=notebook_id)


@notebooks_router.post("/{notebook_id}/sources", response_model=SourceResponse)
async def create_source_for_notebook(
    notebook_id: int,
    payload: SourceCreateRequest,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> SourceResponse:
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=current_user.id
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    source_type = payload.type.strip().lower()
    source_url = payload.url
    if source_type == "url":
        if not source_url or not source_url.strip():
            raise HTTPException(status_code=400, detail="URL is required when source type is 'url'.")
        try:
            source_url = validate_ingestion_url(source_url)
        except URLValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    source = crud.create_source(
        db=db,
        notebook_id=notebook_id,
        source_type=source_type,
        title=payload.title,
        original_name=payload.original_name,
        url=source_url,
        storage_path=payload.storage_path,
        status=payload.status,
    )

    if source_type == "url":
        crud.update_source_status(db=db, source_id=source.id, status="processing")
        try:
            ingested_chunk_count = await run_in_threadpool(
                ingest_source, source=source, owner_user_id=current_user.id
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


@notebooks_router.post("/{notebook_id}/sources/upload", response_model=SourceResponse)
async def upload_source_for_notebook(
    notebook_id: int,
    title: str | None = Form(None),
    status: str = Form("pending"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> SourceResponse:
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=current_user.id
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
            ingest_source, source=source, owner_user_id=current_user.id
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
    notebook_id: int,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> list[SourceResponse]:
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=current_user.id
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
    notebook_id: int,
    payload: ThreadCreateRequest,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> ThreadResponse:
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=current_user.id
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    thread = crud.create_chat_thread(db=db, notebook_id=notebook_id, title=payload.title)
    return ThreadResponse(
        id=thread.id, notebook_id=thread.notebook_id, title=thread.title, created_at=thread.created_at
    )


@notebooks_router.get("/{notebook_id}/threads", response_model=list[ThreadResponse])
def list_threads_for_notebook(
    notebook_id: int,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> list[ThreadResponse]:
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=current_user.id
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
    thread_id: int,
    notebook_id: int,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> list[MessageResponse]:
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=current_user.id
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
    thread_id: int,
    payload: ChatRequest,
    notebook_id: int,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> ChatResponse:
    notebook = crud.get_notebook_for_user(
        db=db, notebook_id=notebook_id, owner_user_id=current_user.id
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    thread = crud.get_thread_for_notebook(db=db, notebook_id=notebook_id, thread_id=thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found for this notebook.")

    prior_messages = crud.list_messages_for_thread(db=db, thread_id=thread_id)
    user_message = crud.create_message(db=db, thread_id=thread_id, role="user", content=payload.question)

    retrieval_rows = query_notebook_chunks(
        owner_user_id=current_user.id,
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


# ── Artifact helpers ──────────────────────────────────────────────────────────

def _artifact_response(artifact) -> ArtifactResponse:
    return ArtifactResponse(
        id=artifact.id,
        notebook_id=artifact.notebook_id,
        type=artifact.type,
        title=artifact.title,
        status=artifact.status,
        content=artifact.content,
        file_path=artifact.file_path,
        metadata=artifact.artifact_metadata,
        error_message=artifact.error_message,
        created_at=artifact.created_at,
        generated_at=artifact.generated_at,
    )


def _run_podcast_background(
    artifact_id: int,
    user_id: int,
    notebook_id: int,
    duration: str,
    topic_focus: str | None,
) -> None:
    """Background task: generate podcast and update the artifact record."""
    db = SessionLocal()
    try:
        crud.update_artifact(db, artifact_id, status="processing")
        generator = PodcastGenerator()
        result = generator.generate_podcast(
            user_id=str(user_id),
            notebook_id=str(notebook_id),
            duration_target=duration,
            topic_focus=topic_focus,
        )
        if "error" in result:
            crud.update_artifact(db, artifact_id, status="failed", error_message=result["error"])
        else:
            transcript_markdown = generator.format_transcript_markdown(result)
            transcript_path = generator.save_transcript(result, str(user_id), str(notebook_id))
            audio_path = result.get("audio_path")
            crud.update_artifact(
                db,
                artifact_id,
                status="ready",
                content=transcript_markdown,
                file_path=(str(audio_path) if audio_path else None),
                metadata={
                    "audio_path": (str(audio_path) if audio_path else None),
                    "transcript_path": transcript_path,
                    **(
                        result.get("metadata", {})
                        if isinstance(result.get("metadata"), dict)
                        else {}
                    ),
                },
            )
    except Exception as exc:
        crud.update_artifact(db, artifact_id, status="failed", error_message=str(exc))
    finally:
        db.close()


# ── Artifact endpoints ────────────────────────────────────────────────────────

@notebooks_router.post("/{notebook_id}/artifacts/report", response_model=ArtifactResponse)
async def generate_report_for_notebook(
    notebook_id: int,
    payload: ReportGenerateRequest,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> ArtifactResponse:
    notebook = crud.get_notebook_for_user(
        db=db,
        notebook_id=notebook_id,
        owner_user_id=current_user.id,
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    detail_level = payload.detail_level.strip().lower()
    if detail_level not in {"short", "medium", "long"}:
        raise HTTPException(status_code=400, detail="detail_level must be one of: short, medium, long")

    artifact = crud.create_artifact(
        db=db,
        notebook_id=notebook_id,
        artifact_type="report",
        title=payload.title or f"Report – {detail_level}",
        metadata={
            "detail_level": detail_level,
            "topic_focus": payload.topic_focus,
        },
    )
    crud.update_artifact(db, artifact.id, status="processing")

    try:
        generator = ReportGenerator()
        result = await run_in_threadpool(
            generator.generate_report,
            user_id=str(current_user.id),
            notebook_id=str(notebook_id),
            detail_level=detail_level,
            topic_focus=payload.topic_focus,
        )
    except Exception as exc:
        crud.update_artifact(db, artifact.id, status="failed", error_message=str(exc))
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}") from exc

    if "error" in result:
        artifact = crud.update_artifact(db, artifact.id, status="failed", error_message=result["error"])
    else:
        content = str(result.get("content", "")).strip()
        report_path = generator.save_report(content, str(current_user.id), str(notebook_id))
        artifact = crud.update_artifact(
            db,
            artifact.id,
            status="ready",
            content=content,
            file_path=report_path,
        )

    return _artifact_response(artifact)


@notebooks_router.post("/{notebook_id}/artifacts/quiz", response_model=ArtifactResponse)
async def generate_quiz_for_notebook(
    notebook_id: int,
    payload: QuizGenerateRequest,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> ArtifactResponse:
    notebook = crud.get_notebook_for_user(
        db=db,
        notebook_id=notebook_id,
        owner_user_id=current_user.id,
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    artifact = crud.create_artifact(
        db=db,
        notebook_id=notebook_id,
        artifact_type="quiz",
        title=payload.title or f"Quiz – {payload.difficulty} ({payload.num_questions}q)",
        metadata={
            "num_questions": payload.num_questions,
            "difficulty": payload.difficulty,
            "topic_focus": payload.topic_focus,
        },
    )
    crud.update_artifact(db, artifact.id, status="processing")

    try:
        generator = QuizGenerator()
        result = await run_in_threadpool(
            generator.generate_quiz,
            user_id=str(current_user.id),
            notebook_id=str(notebook_id),
            num_questions=payload.num_questions,
            difficulty=payload.difficulty,
            topic_focus=payload.topic_focus,
        )
    except Exception as exc:
        crud.update_artifact(db, artifact.id, status="failed", error_message=str(exc))
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {exc}") from exc

    if "error" in result:
        artifact = crud.update_artifact(db, artifact.id, status="failed", error_message=result["error"])
    else:
        quiz_markdown = generator.format_quiz_markdown(result, title=payload.title or "Quiz")
        quiz_path = generator.save_quiz(quiz_markdown, str(current_user.id), str(notebook_id))
        artifact = crud.update_artifact(
            db,
            artifact.id,
            status="ready",
            content=quiz_markdown,
            file_path=quiz_path,
        )

    return _artifact_response(artifact)


@notebooks_router.post("/{notebook_id}/artifacts/podcast", response_model=ArtifactResponse)
def generate_podcast_for_notebook(
    notebook_id: int,
    payload: PodcastGenerateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> ArtifactResponse:
    notebook = crud.get_notebook_for_user(
        db=db,
        notebook_id=notebook_id,
        owner_user_id=current_user.id,
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    artifact = crud.create_artifact(
        db=db,
        notebook_id=notebook_id,
        artifact_type="podcast",
        title=payload.title or f"Podcast – {payload.duration}",
        metadata={
            "duration": payload.duration,
            "topic_focus": payload.topic_focus,
        },
    )

    background_tasks.add_task(
        _run_podcast_background,
        artifact_id=artifact.id,
        user_id=current_user.id,
        notebook_id=notebook_id,
        duration=payload.duration,
        topic_focus=payload.topic_focus,
    )

    return _artifact_response(artifact)


@notebooks_router.get("/{notebook_id}/artifacts", response_model=list[ArtifactResponse])
def list_artifacts_for_notebook(
    notebook_id: int,
    artifact_type: str | None = None,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> list[ArtifactResponse]:
    notebook = crud.get_notebook_for_user(
        db=db,
        notebook_id=notebook_id,
        owner_user_id=current_user.id,
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    artifacts = crud.list_artifacts(db=db, notebook_id=notebook_id, artifact_type=artifact_type)
    return [_artifact_response(a) for a in artifacts]


@notebooks_router.get("/{notebook_id}/artifacts/{artifact_id}", response_model=ArtifactResponse)
def get_artifact_for_notebook(
    notebook_id: int,
    artifact_id: int,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> ArtifactResponse:
    notebook = crud.get_notebook_for_user(
        db=db,
        notebook_id=notebook_id,
        owner_user_id=current_user.id,
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    artifact = crud.get_artifact(db=db, artifact_id=artifact_id)
    if artifact is None or artifact.notebook_id != notebook_id:
        raise HTTPException(status_code=404, detail="Artifact not found.")

    return _artifact_response(artifact)


@notebooks_router.get("/{notebook_id}/artifacts/{artifact_id}/audio")
def download_podcast_audio(
    notebook_id: int,
    artifact_id: int,
    db: Session = Depends(get_db),
    current_user: CurrentUser = Depends(require_current_user),
) -> FileResponse:
    notebook = crud.get_notebook_for_user(
        db=db,
        notebook_id=notebook_id,
        owner_user_id=current_user.id,
    )
    if notebook is None:
        raise HTTPException(status_code=404, detail="Notebook not found for this user.")

    artifact = crud.get_artifact(db=db, artifact_id=artifact_id)
    if artifact is None or artifact.notebook_id != notebook_id:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    if artifact.type != "podcast":
        raise HTTPException(status_code=400, detail="Artifact is not a podcast.")
    if artifact.status != "ready":
        raise HTTPException(status_code=409, detail=f"Podcast not ready yet (status: {artifact.status}).")
    if not artifact.file_path or not Path(artifact.file_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not found on disk.")

    return FileResponse(
        path=artifact.file_path,
        media_type="audio/mpeg",
        filename=Path(artifact.file_path).name,
    )


app.include_router(auth_router)
app.include_router(notebooks_router)
app.include_router(sources_router)
app.include_router(threads_router)
