from __future__ import annotations
import uuid
from typing import List, Dict
import nltk
from transformers import AutoTokenizer
import re


try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt")


def get_tokenizer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def chunk_text(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50,
) -> List[Dict]:
    """Chunk `text` into token-aware overlapping chunks.

    This implementation approximates token counts by tokenizing per-sentence
    and accumulating sentences until chunk size is reached. Returns a list
    of chunk dicts with `chunk_id`, `text`, `char_start`, `char_end`, and `text_preview`.
    """
    tokenizer = get_tokenizer(model_name)
    sentences = nltk.sent_tokenize(text)
    # Fallback: if tokenization didn't split (e.g. sentences glued without spaces)
    if len(sentences) == 1 and re.search(r"\.\w", text):
        fallback = re.sub(r"\.(?=[A-Za-z0-9])", ". ", text)
        sentences = nltk.sent_tokenize(fallback)

    chunks = []
    cur_sents = []
    cur_token_count = 0
    search_pos = 0

    def tokens_for(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    for sent in sentences:
        tcount = tokens_for(sent)
        if cur_token_count + tcount > chunk_size_tokens and cur_sents:
            chunk_text = " ".join(cur_sents).strip()
            # find char offsets (best-effort)
            start = text.find(chunk_text, search_pos)
            if start == -1:
                start = search_pos
            end = start + len(chunk_text)
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "char_start": start,
                    "char_end": end,
                    "text_preview": chunk_text[:200],
                }
            )
            search_pos = end
            # prepare overlap: keep last sentences approximating overlap_tokens
            overlap_sents = []
            overlap_count = 0
            # iterate from the end of cur_sents backwards
            for s in reversed(cur_sents):
                sc = tokens_for(s)
                if overlap_count + sc > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_count += sc
            cur_sents = overlap_sents.copy()
            cur_token_count = overlap_count

        cur_sents.append(sent)
        cur_token_count += tcount

    # final chunk
    if cur_sents:
        chunk_text = " ".join(cur_sents).strip()
        start = text.find(chunk_text, search_pos)
        if start == -1:
            start = search_pos
        end = start + len(chunk_text)
        chunks.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text,
                "char_start": start,
                "char_end": end,
                "text_preview": chunk_text[:200],
            }
        )

    return chunks
