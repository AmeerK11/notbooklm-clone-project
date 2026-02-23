from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from data.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(255))
    avatar_url: Mapped[str | None] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    notebooks: Mapped[list[Notebook]] = relationship(
        "Notebook", back_populates="owner", cascade="all, delete-orphan"
    )


class Notebook(Base):
    __tablename__ = "notebooks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    owner_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    owner: Mapped[User] = relationship("User", back_populates="notebooks")
    sources: Mapped[list[Source]] = relationship(
        "Source", back_populates="notebook", cascade="all, delete-orphan"
    )
    chat_threads: Mapped[list[ChatThread]] = relationship(
        "ChatThread", back_populates="notebook", cascade="all, delete-orphan"
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        "Artifact", back_populates="notebook", cascade="all, delete-orphan"
    )


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    notebook_id: Mapped[int] = mapped_column(
        ForeignKey("notebooks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255))
    original_name: Mapped[str | None] = mapped_column(String(1024))
    url: Mapped[str | None] = mapped_column(String(2048))
    storage_path: Mapped[str | None] = mapped_column(String(1024))
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    notebook: Mapped[Notebook] = relationship("Notebook", back_populates="sources")
    citations: Mapped[list[MessageCitation]] = relationship(
        "MessageCitation", back_populates="source", cascade="all, delete-orphan"
    )


class ChatThread(Base):
    __tablename__ = "chat_threads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    notebook_id: Mapped[int] = mapped_column(
        ForeignKey("notebooks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    notebook: Mapped[Notebook] = relationship("Notebook", back_populates="chat_threads")
    messages: Mapped[list[Message]] = relationship(
        "Message", back_populates="thread", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    thread_id: Mapped[int] = mapped_column(
        ForeignKey("chat_threads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    thread: Mapped[ChatThread] = relationship("ChatThread", back_populates="messages")
    citations: Mapped[list[MessageCitation]] = relationship(
        "MessageCitation", back_populates="message", cascade="all, delete-orphan"
    )


class MessageCitation(Base):
    __tablename__ = "message_citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_id: Mapped[int] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_ref: Mapped[str | None] = mapped_column(String(255))
    quote: Mapped[str | None] = mapped_column(Text)
    score: Mapped[float | None] = mapped_column(Float)

    message: Mapped[Message] = relationship("Message", back_populates="citations")
    source: Mapped[Source] = relationship("Source", back_populates="citations")

class Artifact(Base):
    """Generated artifacts: quizzes, podcasts, reports."""
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    notebook_id: Mapped[int] = mapped_column(
        ForeignKey("notebooks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'quiz', 'podcast', 'report'
    title: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    file_path: Mapped[str | None] = mapped_column(String(1024))
    artifact_metadata: Mapped[dict | None] = mapped_column("metadata", JSON)
    content: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    notebook: Mapped[Notebook] = relationship("Notebook", back_populates="artifacts")