from __future__ import annotations
import argparse
from pathlib import Path
import uuid
import sys
import os
from tqdm import tqdm
from .storage import LocalStorageAdapter
from .extractors import extract_text_from_txt, extract_text_from_url, extract_text_from_pdf, extract_text_from_pptx
from .chunker import chunk_text
from .embeddings import EmbeddingAdapter
from .vectorstore import ChromaAdapter


_EXTRACTORS = {
    ".txt": extract_text_from_txt,
    ".pdf": extract_text_from_pdf,
    ".pptx": extract_text_from_pptx,
}


def handle_upload(args: argparse.Namespace, adapter: LocalStorageAdapter):
    """Upload, extract, and optionally ingest a file."""
    try:
        path = Path(args.path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        source_id = args.source_id or str(uuid.uuid4())
        print(f"[*] Uploading {path.name} (source_id={source_id})...")
        
        dest = adapter.save_raw_file(args.user, args.notebook, source_id, path)
        print(f"[✓] Saved raw file to: {dest}")
        
        # Extract based on extension
        ext = path.suffix.lower()
        if ext not in _EXTRACTORS:
            print(f"[!] No extractor for {ext}. Raw file saved.")
            return
        
        print(f"[*] Extracting text from {ext}...")
        extractor = _EXTRACTORS[ext]
        
        if ext == ".pdf":
            use_ocr = args.ocr if hasattr(args, "ocr") else False
            result = extractor(path, use_ocr=use_ocr)
        else:
            result = extractor(path)
        
        text = result.get("text", "")
        if not text.strip():
            print(f"[!] No text extracted from {path.name}.")
            return
        
        adapter.save_extracted_text(args.user, args.notebook, source_id, "content", text)
        print(f"[✓] Extracted text saved ({len(text)} chars) for source {source_id}")
        
        # Auto-ingest if requested
        if hasattr(args, "auto_ingest") and args.auto_ingest:
            print(f"[*] Auto-ingesting into Chroma...")
            _do_ingest(args.user, args.notebook, source_id, adapter, args)
        else:
            print(f"[>] To ingest: python -m src.ingestion.cli ingest --user {args.user} --notebook {args.notebook} --source-id {source_id}")
    
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}", file=sys.stderr)
        raise SystemExit(1)


def handle_url(args: argparse.Namespace, adapter: LocalStorageAdapter):
    """Fetch, extract, and optionally ingest from a URL."""
    try:
        source_id = args.source_id or str(uuid.uuid4())
        print(f"[*] Fetching and extracting from {args.url}...")
        
        result = extract_text_from_url(args.url)
        text = result.get("text", "")
        if not text.strip():
            print(f"[!] No text extracted from {args.url}.")
            return
        
        nb = adapter.ensure_notebook(args.user, args.notebook)
        raw_dir = nb / "files_raw" / source_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / "page.html"
        raw_path.write_text(result.get("html", ""), encoding="utf-8")
        print(f"[✓] Saved raw HTML to: {raw_path}")
        
        adapter.save_extracted_text(args.user, args.notebook, source_id, "content", text)
        print(f"[✓] Extracted text saved ({len(text)} chars) for source {source_id}")
        
        # Auto-ingest if requested
        if hasattr(args, "auto_ingest") and args.auto_ingest:
            print(f"[*] Auto-ingesting into Chroma...")
            _do_ingest(args.user, args.notebook, source_id, adapter, args)
        else:
            print(f"[>] To ingest: python -m src.ingestion.cli ingest --user {args.user} --notebook {args.notebook} --source-id {source_id}")
    
    except Exception as e:
        print(f"[ERROR] URL extraction failed: {e}", file=sys.stderr)
        raise SystemExit(1)


def _do_ingest(user: str, notebook: str, source_id: str, adapter: LocalStorageAdapter, args: argparse.Namespace):
    """Internal helper: chunk, embed, and ingest into Chroma."""
    try:
        nb = adapter.ensure_notebook(user, notebook)
        extracted_path = nb / "files_extracted" / source_id / "content.txt"
        
        if not extracted_path.exists():
            raise FileNotFoundError(f"Extracted content not found: {extracted_path}")
        
        print(f"[*] Loading extracted text from {source_id}...")
        text = extracted_path.read_text(encoding="utf-8")
        text_len = len(text)
        if not text.strip():
            raise ValueError(f"Source {source_id} has no content.")
        
        print(f"[*] Chunking text ({text_len} chars)...")
        chunk_model = getattr(args, "chunk_model", None) or "sentence-transformers/all-MiniLM-L6-v2"
        chunks = chunk_text(text, model_name=chunk_model)
        
        for c in chunks:
            c["source_id"] = source_id
            c["page"] = None
        
        print(f"[✓] Created {len(chunks)} chunks")
        
        # Initialize embedder with provider switching
        provider = getattr(args, "embedding_provider", None) or os.getenv("EMBEDDING_PROVIDER", "local")
        model_name = getattr(args, "embedding_model", None) or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        
        print(f"[*] Computing embeddings (provider={provider}, model={model_name})...")
        embedder = EmbeddingAdapter(model_name=model_name, provider=provider)
        texts = [c["text"] for c in chunks]
        
        embeddings = []
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
            batch = texts[i : i + batch_size]
            embeddings.extend(embedder.embed_texts(batch, batch_size=len(batch)))
        
        print(f"[✓] Computed {len(embeddings)} embeddings")
        
        print(f"[*] Upserting to Chroma...")
        chroma_dir = str((nb / "chroma").resolve())
        store = ChromaAdapter(persist_directory=chroma_dir)
        store.upsert_chunks(user, notebook, chunks, embeddings)
        
        print(f"[✓] Ingested {len(chunks)} chunks into Chroma collection '{user}_{notebook}'")
    
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}", file=sys.stderr)
        raise SystemExit(1)


def handle_ingest(args: argparse.Namespace, adapter: LocalStorageAdapter):
    """Chunk, embed, and ingest into Chroma."""
    _do_ingest(args.user, args.notebook, args.source_id, adapter, args)


def main():
    p = argparse.ArgumentParser(
        description="NotebookLM-style ingestion CLI: upload, extract, chunk, embed, and store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload and extract (manual 2-step workflow):
  python -m src.ingestion.cli upload --user alice --notebook nb1 --path docs/notes.txt
  python -m src.ingestion.cli ingest --user alice --notebook nb1 --source-id <id>
  
  # Upload and auto-ingest (one-shot workflow):
  python -m src.ingestion.cli upload --user alice --notebook nb1 --path docs/notes.txt --auto-ingest
  
  # URL extraction and auto-ingest:
  python -m src.ingestion.cli url --user alice --notebook nb1 --url https://example.com --auto-ingest
  
  # Ingest with custom embedding provider:
  python -m src.ingestion.cli ingest --user alice --notebook nb1 --source-id <id> \\
    --embedding-provider openai --embedding-model text-embedding-3-large
        """,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Upload command
    up = sub.add_parser("upload", help="Upload and extract a file")
    up.add_argument("--user", required=True, help="User ID")
    up.add_argument("--notebook", required=True, help="Notebook ID")
    up.add_argument("--path", required=True, help="Path to file (*.txt, *.pdf, *.pptx)")
    up.add_argument("--source-id", required=False, help="Source ID (auto-generated if omitted)")
    up.add_argument("--ocr", action="store_true", help="(PDF only) Enable OCR on images")
    up.add_argument("--auto-ingest", action="store_true", help="Automatically chunk, embed, and ingest into Chroma")

    # URL command
    urlp = sub.add_parser("url", help="Extract text from a URL")
    urlp.add_argument("--user", required=True, help="User ID")
    urlp.add_argument("--notebook", required=True, help="Notebook ID")
    urlp.add_argument("--url", required=True, help="URL to fetch")
    urlp.add_argument("--source-id", required=False, help="Source ID (auto-generated if omitted)")
    urlp.add_argument("--auto-ingest", action="store_true", help="Automatically chunk, embed, and ingest into Chroma")

    # Ingest command
    ingp = sub.add_parser("ingest", help="Chunk, embed, and ingest into Chroma")
    ingp.add_argument("--user", required=True, help="User ID")
    ingp.add_argument("--notebook", required=True, help="Notebook ID")
    ingp.add_argument("--source-id", required=True, help="Source ID (from upload/url)")
    ingp.add_argument(
        "--embedding-provider",
        choices=["local", "openai", "huggingface"],
        default="local",
        help="Embedding provider (default: local). Set API keys via env vars.",
    )
    ingp.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)",
    )
    ingp.add_argument(
        "--chunk-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Tokenizer model for chunking (default: all-MiniLM-L6-v2)",
    )

    args = p.parse_args()
    
    try:
        adapter = LocalStorageAdapter()
        
        if args.cmd == "upload":
            handle_upload(args, adapter)
        elif args.cmd == "url":
            handle_url(args, adapter)
        elif args.cmd == "ingest":
            handle_ingest(args, adapter)
    except KeyboardInterrupt:
        print("\n[!] Cancelled by user.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
