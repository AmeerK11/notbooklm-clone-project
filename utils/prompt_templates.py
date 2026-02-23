from __future__ import annotations


def build_rag_system_prompt() -> str:
    return (
        "You are a precise research assistant. Answer only using the provided context. "
        "Use conversation history only to resolve references (for example: 'that', 'it', 'previous point'). "
        "If context is insufficient, say you do not have enough information. "
        "Keep responses concise and factual."
    )


def build_rag_user_prompt(
    question: str,
    context_blocks: list[str],
    conversation_history: list[str] | None = None,
) -> str:
    joined_context = "\n\n".join(context_blocks)
    joined_history = "\n".join(conversation_history or [])
    return (
        "Conversation history (oldest -> newest):\n"
        f"{joined_history or '[No prior messages]'}\n\n"
        "Context:\n"
        f"{joined_context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Instructions:\n"
        "1) Answer from context only.\n"
        "2) If missing evidence, explicitly say so.\n"
        "3) End with a short 'Sources used' section referencing source titles first, then chunk identifiers."
    )
