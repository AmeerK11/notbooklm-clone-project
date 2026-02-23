from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


class LLMConfigError(RuntimeError):
    pass


def _get_client() -> Groq:
    if not _GROQ_API_KEY:
        raise LLMConfigError("GROQ_API_KEY is not set.")
    return Groq(api_key=_GROQ_API_KEY)


def generate_chat_completion(system_prompt: str, user_prompt: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""
