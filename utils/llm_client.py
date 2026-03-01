from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class LLMConfigError(RuntimeError):
    pass


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise LLMConfigError("GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)


def generate_chat_completion(system_prompt: str, user_prompt: str) -> str:
    client = _get_client()
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""
