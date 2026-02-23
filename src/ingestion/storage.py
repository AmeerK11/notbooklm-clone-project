from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import shutil
import json
import uuid
from datetime import datetime


class StorageAdapter:
    """Abstract storage adapter. Implementations must provide these methods.
    """

    def ensure_notebook(self, user_id: str, notebook_id: str) -> Path:
        raise NotImplementedError()

    def save_raw_file(self, user_id: str, notebook_id: str, source_id: str, src_path: Path) -> Path:
        raise NotImplementedError()

    def save_extracted_text(self, user_id: str, notebook_id: str, source_id: str, filename: str, text: str) -> Path:
        raise NotImplementedError()

    def read_index(self, user_id: str) -> Dict[str, Any]:
        raise NotImplementedError()

    def write_index(self, user_id: str, index: Dict[str, Any]) -> None:
        raise NotImplementedError()


class LocalStorageAdapter(StorageAdapter):
    """Local filesystem storage adapter following the project's `data/` layout.

    Example usage:
        adapter = LocalStorageAdapter(base_dir="data")
        adapter.ensure_notebook("alice", "nb-123")

    """

    def __init__(self, base_dir: str = "data"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def _user_dir(self, user_id: str) -> Path:
        return self.base / "users" / user_id

    def _notebooks_dir(self, user_id: str) -> Path:
        return self._user_dir(user_id) / "notebooks"

    def ensure_notebook(self, user_id: str, notebook_id: str) -> Path:
        notebooks = self._notebooks_dir(user_id)
        notebooks.mkdir(parents=True, exist_ok=True)
        nb_dir = notebooks / notebook_id
        nb_dir.mkdir(parents=True, exist_ok=True)
        # create subfolders
        (nb_dir / "files_raw").mkdir(exist_ok=True)
        (nb_dir / "files_extracted").mkdir(exist_ok=True)
        (nb_dir / "chroma").mkdir(exist_ok=True)
        (nb_dir / "chat").mkdir(exist_ok=True)
        (nb_dir / "artifacts").mkdir(exist_ok=True)
        # ensure per-user index exists
        idx = self._user_dir(user_id) / "notebooks" / "index.json"
        if not idx.exists():
            idx.parent.mkdir(parents=True, exist_ok=True)
            idx.write_text(json.dumps({"notebooks": []}, indent=2), encoding="utf-8")
        # register notebook in index if missing
        self._register_notebook_in_index(user_id, notebook_id)
        return nb_dir

    def _register_notebook_in_index(self, user_id: str, notebook_id: str):
        idx_path = self._user_dir(user_id) / "notebooks" / "index.json"
        try:
            data = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            data = {"notebooks": []}
        known = {n.get("id") for n in data.get("notebooks", [])}
        if notebook_id not in known:
            data.setdefault("notebooks", []).append({
                "id": notebook_id,
                "name": notebook_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
            })
            idx_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save_raw_file(self, user_id: str, notebook_id: str, source_id: str, src_path: Path) -> Path:
        nb = self.ensure_notebook(user_id, notebook_id)
        dest_dir = nb / "files_raw" / source_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src_path.name
        shutil.copy2(src_path, dest)
        return dest

    def save_extracted_text(self, user_id: str, notebook_id: str, source_id: str, filename: str, text: str) -> Path:
        nb = self.ensure_notebook(user_id, notebook_id)
        dest_dir = nb / "files_extracted" / source_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{filename}.txt"
        dest.write_text(text, encoding="utf-8")
        return dest

    def read_index(self, user_id: str) -> Dict[str, Any]:
        idx = self._user_dir(user_id) / "notebooks" / "index.json"
        if not idx.exists():
            return {"notebooks": []}
        return json.loads(idx.read_text(encoding="utf-8"))

    def write_index(self, user_id: str, index: Dict[str, Any]) -> None:
        idx = self._user_dir(user_id) / "notebooks" / "index.json"
        idx.parent.mkdir(parents=True, exist_ok=True)
        idx.write_text(json.dumps(index, indent=2), encoding="utf-8")
