# Integration Guide: Ingestion Module

## Overview

The ingestion module (`src/ingestion/`) implements the **complete source ingestion pipeline** for the NotebookLM project. It handles:

- **Source extraction**: PDF, PPTX, TXT files, and web URLs
- **Text chunking**: Token-aware sentence-based chunking
- **Embedding**: Local (offline) or cloud-based (OpenAI, HuggingFace) embeddings
- **Vector storage**: Persistent Chroma database per user/notebook
- **CLI interface**: For testing and direct integration

## Architecture

```
┌─────────────┐     ┌────────────┐     ┌──────────┐     ┌────────┐
│   Upload    │────▶│  Extract   │────▶│  Chunk   │────▶│ Embed  │
│  File/URL   │     │   Text     │     │  Text    │     │ Text   │
└─────────────┘     └────────────┘     └──────────┘     └────────┘
                                                             │
                                                             ▼
                                                        ┌──────────┐
                                                        │  Chroma  │
                                                        │ VectorDB │
                                                        └──────────┘
```

**Module Files:**
- `extractors.py`: Text extraction for multiple formats
- `chunker.py`: Token-aware chunking with NLTK
- `embeddings.py`: Embedding provider abstraction (local/OpenAI/HuggingFace)
- `vectorstore.py`: Chroma wrapper with per-user/notebook isolation
- `storage.py`: File system storage adapter
- `cli.py`: CLI interface for testing

## Storage Structure

Data is stored in a per-user, per-notebook structure:

```
data/
└── users/
    └── {username}/
        └── notebooks/
            └── {notebook-uuid}/
                ├── files_raw/                    ← Original uploaded files
                │   └── {source-id}/
                │       └── {filename}
                ├── files_extracted/              ← Plain text extracted from sources
                │   └── {source-id}/
                │       └── content.txt
                ├── chroma/                       ← Vector database
                │   ├── chroma.sqlite3
                │   └── {collection-uuid}/
                │       ├── data_level0.bin
                │       ├── header.bin
                │       ├── length.bin
                │       └── link_lists.bin
                ├── chat/                         ← (For RAG chat module)
                │   └── messages.jsonl
                └── artifacts/                    ← (For artifact generation module)
                    ├── reports/
                    ├── quizzes/
                    └── podcasts/
```

## Core APIs

### 1. Ingest Sources (CLI)

**Option A: Upload file with auto-ingest (recommended for UI)**
```bash
python -m src.ingestion.cli upload \
  --user alice \
  --notebook notebook-123 \
  --path /path/to/document.pdf \
  --auto-ingest
```

**Option B: Upload + extract, then ingest separately**
```bash
# Step 1: Upload and extract
python -m src.ingestion.cli upload \
  --user alice \
  --notebook notebook-123 \
  --path /path/to/document.pdf

# Step 2: Ingest extracted content
python -m src.ingestion.cli ingest \
  --user alice \
  --notebook notebook-123 \
  --source-id <source-id-from-step-1>
```

**Option C: Ingest from URL with auto-ingest**
```bash
python -m src.ingestion.cli url \
  --user alice \
  --notebook notebook-123 \
  --url https://example.com/article \
  --auto-ingest
```

### 2. Query Vector Database (RAG)

**Import and use ChromaAdapter directly:**

```python
from src.ingestion.vectorstore import ChromaAdapter
from pathlib import Path

# Initialize adapter
user_id = "alice"
notebook_id = "notebook-123"
chroma_dir = f"data/users/{user_id}/notebooks/{notebook_id}/chroma"

store = ChromaAdapter(persist_directory=chroma_dir)

# Query for similar chunks
query_text = "What is machine learning?"
top_k = 5
results = store.query(user_id, notebook_id, query_text, top_k=top_k)

# Process results
for chunk_id, distance, chunk_data in results:
    source_id = chunk_data["metadata"]["source_id"]
    text = chunk_data["document"]
    print(f"Source: {source_id}")
    print(f"Distance: {distance}")
    print(f"Text: {text}")
    print("---")
```

**ChromaAdapter.query() returns:**
```python
List[Tuple[str, float, Dict[str, Any]]]
# (chunk_id, distance_score, {
#   "document": str,           # Actual text chunk
#   "metadata": {
#     "source_id": str,        # Which source this came from
#     "page": int | None,      # Page number (PDFs)
#     "text_preview": str,     # First 100 chars
#     "char_start": int,       # Position in original
#     "char_end": int          # Position in original
#   }
# })
```

### 3. Pythonic Integration (Programmatic)

If you need to embed without using the CLI:

```python
from src.ingestion.storage import LocalStorageAdapter
from src.ingestion.extractors import extract_text_from_pdf
from src.ingestion.chunker import chunk_text
from src.ingestion.embeddings import EmbeddingAdapter
from src.ingestion.vectorstore import ChromaAdapter
import uuid

user_id = "alice"
notebook_id = "nb-123"
file_path = "path/to/document.pdf"

# Step 1: Extract
adapter = LocalStorageAdapter()
source_id = str(uuid.uuid4())
result = extract_text_from_pdf(file_path, use_ocr=False)
text = result["text"]

# Step 2: Save raw and extracted
adapter.save_raw_file(user_id, notebook_id, source_id, file_path)
adapter.save_extracted_text(user_id, notebook_id, source_id, "content", text)

# Step 3: Chunk
chunks = chunk_text(text, model_name="sentence-transformers/all-MiniLM-L6-v2")
for c in chunks:
    c["source_id"] = source_id
    c["page"] = None

# Step 4: Embed (using provider)
embedder = EmbeddingAdapter(
    model_name="all-MiniLM-L6-v2",
    provider="local"  # or "openai", "huggingface"
)
texts = [c["text"] for c in chunks]
embeddings = embedder.embed_texts(texts, batch_size=32)

# Step 5: Store in Chroma
nb = adapter.ensure_notebook(user_id, notebook_id)
chroma_dir = str((nb / "chroma").resolve())
store = ChromaAdapter(persist_directory=chroma_dir)
store.upsert_chunks(user_id, notebook_id, chunks, embeddings)

print(f"✓ Ingested {len(chunks)} chunks")
```

## Configuration

### Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```bash
# Embedding provider: "local" (default), "openai", or "huggingface"
EMBEDDING_PROVIDER=local

# OpenAI (if provider=openai)
OPENAI_API_KEY=sk-...

# HuggingFace (if provider=huggingface)
HF_API_TOKEN=hf_...

# Model names (optional, defaults shown)
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Storage
DATA_DIR=data/
```

### Embedding Providers

**Local (Recommended for MVP)**
```bash
EMBEDDING_PROVIDER=local
# Downloads: sentence-transformers/all-MiniLM-L6-v2 (~91MB on first run)
# No API keys required, works offline
```

**OpenAI**
```bash
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small  # or text-embedding-3-large
```

**HuggingFace**
```bash
EMBEDDING_PROVIDER=huggingface
HF_API_TOKEN=hf_...
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## For RAG Chat Module

Once sources are ingested, the RAG chat module can:

1. **Accept user query** from frontend
2. **Retrieve context** using `ChromaAdapter.query(user, notebook, query_text, top_k=5)`
3. **Build prompt** with retrieved chunks + user query
4. **Call LLM** (GPT-4, Claude, etc.)
5. **Return answer** with source citations (using source_id + text_preview)

### Example RAG Flow

```python
def rag_chat(user_id: str, notebook_id: str, query: str):
    """Retrieve-augmented chat with citations."""
    
    # 1. Get chroma dir for this user/notebook
    chroma_dir = f"data/users/{user_id}/notebooks/{notebook_id}/chroma"
    store = ChromaAdapter(persist_directory=chroma_dir)
    
    # 2. Retrieve top-5 relevant chunks
    results = store.query(user_id, notebook_id, query, top_k=5)
    
    # 3. Format context with citations
    context = ""
    citations = []
    for i, (chunk_id, distance, chunk_data) in enumerate(results, 1):
        text = chunk_data["document"]
        source_id = chunk_data["metadata"]["source_id"]
        context += f"[{i}] {text}\n\n"
        citations.append({
            "id": i,
            "source_id": source_id,
            "preview": chunk_data["metadata"]["text_preview"]
        })
    
    # 4. Build prompt (with your system prompt)
    prompt = f"""
Context from uploaded sources:
{context}

User question: {query}

Answer based on the context above. If not in context, say so.
    """
    
    # 5. Call LLM (e.g., OpenAI, Claude)
    response = call_llm(prompt)  # Your LLM integration
    
    # 6. Return with citations
    return {
        "answer": response,
        "citations": citations
    }
```

## Testing

All modules are unit and integration tested:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_integration.py -v

# Run specific test
pytest tests/test_integration.py::test_txt_upload_extract_ingest -v
```

Current test coverage:
- ✅ Text extraction (TXT, PDF, PPTX, URL)
- ✅ Chunking with NLTK
- ✅ Embedding (local + provider switching)
- ✅ Chroma isolation by user/notebook
- ✅ End-to-end ingestion pipeline

## Troubleshooting

### "No text extracted"
- **PDF**: Scanned images without OCR → add `--ocr` flag (requires pytesseract)
- **URL**: Try with User-Agent header (handled automatically) or check network access

### "Chroma collection not found"
- Check: data folder exists and has correct user/notebook structure
- Try: Re-ingest sources or check chroma_dir path

### "Embedding provider error"
- **Missing package**: `pip install openai` or `pip install huggingface_hub`
- **Missing API key**: Check `.env` file for OPENAI_API_KEY or HF_API_TOKEN

### "NLTK punkt tokenizer not found"
- Automatically downloaded on first run
- If fails: `python -c "import nltk; nltk.download('punkt_tab')"`

## Development Notes

### Extending Extractors

Add new format support in `src/ingestion/extractors.py`:

```python
def extract_text_from_docx(file_path: Path) -> Dict[str, str]:
    """Extract text from DOCX files."""
    from docx import Document
    doc = Document(file_path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return {"text": text}

# Register in cli.py _EXTRACTORS
_EXTRACTORS = {
    ".txt": extract_text_from_txt,
    ".pdf": extract_text_from_pdf,
    ".pptx": extract_text_from_pptx,
    ".docx": extract_text_from_docx,  # NEW
}
```

### Custom Embedding Models

Pass any HuggingFace sentence-transformer model name:

```bash
python -m src.ingestion.cli ingest \
  --user alice \
  --notebook nb1 \
  --source-id abc123 \
  --embedding-model all-mpnet-base-v2  # Different model
```

### Batch Processing

For bulk ingestion:

```bash
# Process multiple files
for file in documents/*.pdf; do
    python -m src.ingestion.cli upload \
        --user alice \
        --notebook batch-$(date +%s) \
        --path "$file" \
        --auto-ingest
done
```

## Next Steps for Teammates

### Frontend Team
- Import CLI commands into Gradio callback: `subprocess.run(["python", "-m", "src.ingestion.cli", ...])`
- Display upload progress using progress bar output
- Parse source_ids from CLI output for metadata storage

### RAG Chat Team
- Use `ChromaAdapter.query()` to retrieve context
- Implement prompt engineering with citations
- Integrate with LLM (OpenAI, Claude, etc.)

### Artifact Generation Team
- Query chunks for context using `ChromaAdapter`
- Generate reports/quizzes using retrieved sources
- Save outputs to `artifacts/{reports,quizzes,podcasts}/`

### Deployment Team
- Ensure `data/` directory is persistent (not ephemeral)
- Set `EMBEDDING_PROVIDER=local` for HF Spaces (no API costs)
- Pre-download models: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

## Contact & Debugging

If integration issues arise:
1. Check `README.md` for dependency installation
2. Run `pytest tests/ -v` to verify module health
3. Check `.env` file and required API keys
4. Review storage folder structure: `ls -R data/users/`
