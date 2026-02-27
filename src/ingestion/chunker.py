from __future__ import annotations
import uuid
from typing import List, Dict
import nltk
from transformers import AutoTokenizer
import re


try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    pass


def get_tokenizer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    try:
        # Prefer local cache only so ingestion does not require network.
        return AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
    except Exception:
        class _FallbackTokenizer:
            model_max_length = 512

            def encode(self, s: str, add_special_tokens: bool = False):
                # Rough token estimate for offline fallback.
                return s.split()

            def decode(self, token_ids, skip_special_tokens: bool = True):
                return " ".join(str(t) for t in token_ids)

        return _FallbackTokenizer()


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
    model_max = int(getattr(tokenizer, "model_max_length", 512) or 512)
    if model_max <= 0 or model_max > 100000:
        model_max = 512
    # Leave room for special tokens.
    effective_chunk_size = min(chunk_size_tokens, max(32, model_max - 2))
    effective_overlap = min(overlap_tokens, max(0, effective_chunk_size // 2))
    def _safe_sent_tokenize(raw_text: str) -> List[str]:
        try:
            return nltk.sent_tokenize(raw_text)
        except LookupError:
            # punkt not available and offline mode: fallback sentence split.
            split = re.split(r"(?<=[.!?])\s+", raw_text)
            return [s for s in split if s.strip()]

    sentences = _safe_sent_tokenize(text)
    # Fallback: if tokenization didn't split (e.g. sentences glued without spaces)
    if len(sentences) == 1 and re.search(r"\.\w", text):
        fallback = re.sub(r"\.(?=[A-Za-z0-9])", ". ", text)
        sentences = _safe_sent_tokenize(fallback)

    chunks = []
    cur_sents = []
    cur_token_count = 0
    search_pos = 0

    def tokens_for(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    def split_long_sentence(sent: str) -> List[str]:
        token_ids = tokenizer.encode(sent, add_special_tokens=False)
        if len(token_ids) <= effective_chunk_size:
            return [sent]
        pieces: List[str] = []
        stride = max(1, effective_chunk_size - effective_overlap)
        for i in range(0, len(token_ids), stride):
            part_ids = token_ids[i : i + effective_chunk_size]
            if not part_ids:
                continue
            pieces.append(tokenizer.decode(part_ids, skip_special_tokens=True).strip())
            if i + effective_chunk_size >= len(token_ids):
                break
        return [p for p in pieces if p]

    for raw_sent in sentences:
        for sent in split_long_sentence(raw_sent):
            tcount = tokens_for(sent)
            if cur_token_count + tcount > effective_chunk_size and cur_sents:
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
                    if overlap_count + sc > effective_overlap:
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
