import sys
import pathlib
import json

# Ensure `src` is on sys.path so tests can import the package
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ingestion.storage as storage
import ingestion.chunker as chunker


def test_local_storage_adapter_creates_dirs_and_index(tmp_path):
    base = tmp_path / "data"
    adapter = storage.LocalStorageAdapter(base_dir=str(base))
    user = "testuser"
    nb = "nb-test-1"
    nb_dir = adapter.ensure_notebook(user, nb)
    assert (nb_dir / "files_raw").exists()
    assert (nb_dir / "files_extracted").exists()
    idx = base / "users" / user / "notebooks" / "index.json"
    data = json.loads(idx.read_text(encoding="utf-8"))
    assert any(n.get("id") == nb for n in data.get("notebooks", []))


def test_chunker_splits_with_mock_tokenizer(monkeypatch):
    # build a long repetitive text to force multiple chunks
    text = "Sentence one." * 200

    class DummyTokenizer:
        def encode(self, s, add_special_tokens=False):
            # approximate tokens by whitespace-separated words
            return [0] * max(1, len(s.split()))

    # Replace tokenizer factory with dummy to avoid downloading models
    monkeypatch.setattr(chunker, "get_tokenizer", lambda model_name=None: DummyTokenizer())

    chunks = chunker.chunk_text(text, model_name="dummy", chunk_size_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    for c in chunks:
        assert isinstance(c.get("text"), str) and len(c.get("text")) > 0
