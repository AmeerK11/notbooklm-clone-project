# Ingestion Module — NotebookLM Clone (MVP)

This repository contains the ingestion module for a NotebookLM-style project. The ingestion pipeline extracts text from multiple source types, chunks text intelligently, computes embeddings (with provider flexibility), and stores vectors in Chroma for later RAG use.

## Features

- **Multi-format source extraction**: TXT, PDF (with optional OCR via pytesseract), PPTX, and URLs
- **Token-aware intelligent chunking**: Sentence-based splitting with configurable overlap and token limits
- **Flexible embedding providers**: Switch between local (sentence-transformers), OpenAI, and HuggingFace APIs via env vars
- **Local-first by default**: Runs fully offline with no API keys required
- **Structured storage**: File-based raw/extracted organization + Chroma vectors with user/notebook isolation
- **CLI interface**: Simple commands for upload, URL extraction, and end-to-end ingestion
- **Comprehensive testing**: Unit tests + integration tests covering the full pipeline


## Quick Start

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv .venv
# Windows PowerShell:
. .venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Embedding Provider (Optional)

Copy `.env.example` to `.env` and set your preferred provider:

```bash
cp .env.example .env
# Edit .env to choose provider: "local" (default), "openai", or "huggingface"
```

- **Local** (default): Uses sentence-transformers (offline, no API key)
- **OpenAI**: Set `OPENAI_API_KEY` (requires active OpenAI account)
- **HuggingFace**: Set `HF_API_TOKEN` (requires HF account)

### 3. CLI Usage Examples

#### Upload and extract a text file:
```bash
python -m src.ingestion.cli upload --user alice --notebook nb1 --path tests/data/sample.txt
```

#### Extract from a URL:
```bash
python -m src.ingestion.cli url --user alice --notebook nb1 --url https://example.com/article
```

#### Ingest into Chroma (chunk, embed, store):
```bash
python -m src.ingestion.cli ingest --user alice --notebook nb1 --source-id <source-id>
```

#### Ingest with custom embedding provider:
```bash
python -m src.ingestion.cli ingest --user alice --notebook nb1 \
  --source-id <source-id> \
  --embedding-provider openai \
  --embedding-model text-embedding-3-large
```

### 4. Run Tests

```bash
pytest -v   # Verbose
pytest -q   # Quiet
```


## Supported File Types

| Format | Extractor | Notes |
|--------|-----------|-------|
| `.txt` | `extract_text_from_txt()` | UTF-8 or latin-1 encoding |
| `.pdf` | `extract_text_from_pdf()` | Optional OCR with `--ocr` flag (requires pytesseract) |
| `.pptx` | `extract_text_from_pptx()` | Extracts text from all slides |
| URL | `extract_text_from_url()` | Fetches & uses readability for main content |


## Architecture

### Core Modules

- **`src/ingestion/storage.py`**: LocalStorageAdapter for file organization (raw/extracted/chunks)
- **`src/ingestion/extractors.py`**: Multi-format text extraction (TXT, PDF, PPTX, URL)
- **`src/ingestion/chunker.py`**: Token-aware intelligent chunking with NLTK & transformers
- **`src/ingestion/embeddings.py`**: Provider-switching embedding adapter (local/OpenAI/HF)
- **`src/ingestion/vectorstore.py`**: ChromaDB wrapper with user/notebook isolation
- **`src/ingestion/cli.py`**: Full-featured CLI for upload, URL, and ingest operations

### Storage Layout

```
data/
  users/
    <user_id>/
      notebooks/
        <notebook_id>/
          files_raw/              # Original file uploads
          files_extracted/        # Extracted text
          chroma/                 # Persistent Chroma data
```


## Configuration & Environment Variables

See `.env.example` for all options:

```bash
# Embedding configuration
EMBEDDING_PROVIDER=local              # [local|openai|huggingface]
EMBEDDING_MODEL=all-MiniLM-L6-v2     # Model identifier
OPENAI_API_KEY=sk-...                 # For OpenAI provider
HF_API_TOKEN=hf_...                   # For HuggingFace provider

# Storage configuration
STORAGE_BASE_DIR=./data               # Base directory for file storage
CHROMA_PERSIST_DIR=./chroma_data      # Chroma persistence (optional)
```


## Optional Dependencies

For enhanced functionality, install optional packages:

```bash
# PDF with OCR (requires system tesseract installation)
pip install pytesseract pillow pdf2image

# LangChain integration (future)
pip install langchain

# Additional models
pip install openai tiktoken
```


## Testing

```bash
# Run all tests
pytest -v

# Run specific test module
pytest tests/test_storage_and_chunker.py -v

# Run integration tests only
pytest tests/test_integration.py -v

# Check coverage
pytest --cov=src tests/
```


## API Examples

### Python API

```python
from src.ingestion.extractors import extract_text_from_txt, extract_text_from_pdf
from src.ingestion.chunker import chunk_text
from src.ingestion.embeddings import EmbeddingAdapter
from src.ingestion.vectorstore import ChromaAdapter

# Extract text
result = extract_text_from_txt("path/to/file.txt")
text = result["text"]

# Chunk
chunks = chunk_text(text, chunk_size_tokens=500, overlap_tokens=50)

# Embed (with provider switching)
embedder = EmbeddingAdapter(provider="local", model_name="all-MiniLM-L6-v2")
embeddings = embedder.embed_texts([c["text"] for c in chunks])

# Store in Chroma
store = ChromaAdapter(persist_directory="./data/chroma")
store.upsert_chunks("alice", "notebook1", chunks, embeddings)
```


## Notes

- **Default stack is local-first** — no API keys required. All processing happens offline using sentence-transformers.
- **PDF OCR**: Requires system `tesseract` installation. See [pytesseract docs](https://github.com/madmaze/pytesseract) for setup.
- **Chunking**: Token counts approximate document length. Adjust `chunk_size_tokens` and `overlap_tokens` for your use case.
- **Embedding dimensions**: all-MiniLM-L6-v2 produces 384-dim vectors. OpenAI text-embedding-3-small produces 1536-dim.
- **Chroma persistence**: Uses DuckDB+Parquet backend when `persist_directory` is set. Ephemeral (in-memory) mode for testing.

1. Install Python 3.10.19, create a virtual environment, and install dependencies:

```bash
# install Python 3.10.11 (use installer from python.org or your package manager)
python --version  # should report 3.10.11
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
. .venv\Scripts\Activate.ps1
# then install dependencies
pip install -r requirements.txt
```

2. CLI usage examples (from repo root):

- Upload a text file (saves raw file and extracts text for .txt files):

```bash
python -m src.ingestion.cli upload --user alice --notebook nb1 --path tests/data/sample.txt
```

- Ingest an extracted source into Chroma (run after upload/url):

```bash
# supply the source-id printed during upload or omit to let CLI create one
python -m src.ingestion.cli ingest --user alice --notebook nb1 --source-id <source_id>
```

3. Run tests:

```bash
pytest -q
```

Files of interest

- `src/ingestion/storage.py`: LocalStorageAdapter and storage layout.
- `src/ingestion/extractors.py`: TXT and URL extractors.
- `src/ingestion/chunker.py`: Token-aware chunker.
- `src/ingestion/embeddings.py`: Local sentence-transformers adapter.
- `src/ingestion/vectorstore.py`: Chroma adapter.
- `src/ingestion/cli.py`: Simple CLI to exercise upload, url, and ingest flows.

Notes

- Default stack is local-first (no API keys required). If you enable OpenAI/HF embedding providers or cloud storage, set `OPENAI_API_KEY`, `HF_API_TOKEN`, or cloud credentials as appropriate.
- For large PDFs requiring OCR, install `tesseract` and the optional Python packages listed in `requirements.txt` comment section.
