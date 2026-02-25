import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import config


class DocumentIndexManager:
    def __init__(self, index_path: str = config.DOCUMENT_INDEX_PATH):
        self._index_path = Path(index_path)
        self._lock = threading.Lock()
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._index_path.exists():
            self._write(self._empty_state())

    @staticmethod
    def _empty_state() -> Dict:
        return {"documents": {}, "kbs": {}}

    def _normalize_state(self, data: Dict) -> Dict:
        normalized = data if isinstance(data, dict) else {}
        normalized.setdefault("documents", {})
        normalized.setdefault("kbs", {})
        return normalized

    def _read(self) -> Dict:
        raw = self._index_path.read_text(encoding="utf-8")
        parsed = json.loads(raw) if raw else self._empty_state()
        return self._normalize_state(parsed)

    def _write(self, data: Dict) -> None:
        self._index_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def ensure_kb(self, kb_id: str, touch_updated: bool = False) -> Dict:
        with self._lock:
            data = self._read()
            kbs = data.setdefault("kbs", {})
            existing = kbs.get(kb_id)
            now = self._now()
            created_at = existing.get("created_at") if isinstance(existing, dict) else None
            existing_updated_at = existing.get("updated_at") if isinstance(existing, dict) else None
            record = {
                "kb_id": kb_id,
                "created_at": created_at or now,
                "updated_at": now if (touch_updated or not existing_updated_at) else existing_updated_at,
            }
            kbs[kb_id] = record
            self._write(data)
            return record

    def upsert_document(
        self,
        kb_id: str,
        document_id: str,
        source_name: str,
        markdown_path: str,
        parent_ids: List[str],
        child_count: int,
        metadata: Optional[Dict] = None,
        dedupe_key: Optional[str] = None,
    ) -> Dict:
        with self._lock:
            data = self._read()
            data.setdefault("documents", {})
            data.setdefault("kbs", {})
            now = self._now()
            record = {
                "kb_id": kb_id,
                "document_id": document_id,
                "source_name": source_name,
                "markdown_path": markdown_path,
                "parent_ids": list(parent_ids),
                "parent_count": len(parent_ids),
                "child_count": int(child_count),
                "metadata": metadata or {},
                "dedupe_key": dedupe_key,
                "created_at": now,
            }
            data["documents"][document_id] = record
            kb_existing = data["kbs"].get(kb_id, {})
            data["kbs"][kb_id] = {
                "kb_id": kb_id,
                "created_at": kb_existing.get("created_at") or now,
                "updated_at": now,
            }
            self._write(data)
            return record

    def get_document(self, kb_id: str, document_id: str) -> Optional[Dict]:
        with self._lock:
            data = self._read()
            record = data.get("documents", {}).get(document_id)
            if not record or record.get("kb_id") != kb_id:
                return None
            return record

    def list_documents(self, kb_id: str) -> List[Dict]:
        with self._lock:
            data = self._read()
            docs = [
                v
                for v in data.get("documents", {}).values()
                if v.get("kb_id") == kb_id
            ]
            docs.sort(key=lambda x: x.get("created_at", ""))
            return docs

    def list_kbs(self) -> List[Dict]:
        with self._lock:
            data = self._read()
            docs = data.get("documents", {})
            kb_meta = data.get("kbs", {})

            grouped_docs: Dict[str, List[Dict]] = {}
            for doc in docs.values():
                kb = doc.get("kb_id")
                if not kb:
                    continue
                grouped_docs.setdefault(kb, []).append(doc)

            all_kb_ids = set(grouped_docs.keys()) | set(kb_meta.keys())
            rows: List[Dict] = []
            for kb_id in sorted(all_kb_ids):
                kb_docs = grouped_docs.get(kb_id, [])
                kb_info = kb_meta.get(kb_id, {}) if isinstance(kb_meta.get(kb_id), dict) else {}

                latest_doc_ts = max((d.get("created_at") or "" for d in kb_docs), default="")
                updated_candidates = [kb_info.get("updated_at") or "", latest_doc_ts]
                last_updated_at = max(updated_candidates) if any(updated_candidates) else None

                rows.append(
                    {
                        "kb_id": kb_id,
                        "document_count": len(kb_docs),
                        "last_updated_at": last_updated_at,
                        "created_at": kb_info.get("created_at"),
                    }
                )

            return rows

    def find_by_dedupe_key(self, kb_id: str, dedupe_key: str) -> Optional[Dict]:
        if not dedupe_key:
            return None
        with self._lock:
            data = self._read()
            for doc in data.get("documents", {}).values():
                if doc.get("kb_id") == kb_id and doc.get("dedupe_key") == dedupe_key:
                    return doc
            return None

    def remove_document(self, kb_id: str, document_id: str) -> Optional[Dict]:
        with self._lock:
            data = self._read()
            doc = data.get("documents", {}).get(document_id)
            if not doc or doc.get("kb_id") != kb_id:
                return None
            del data["documents"][document_id]
            kb_existing = data.get("kbs", {}).get(kb_id, {})
            data.setdefault("kbs", {})[kb_id] = {
                "kb_id": kb_id,
                "created_at": kb_existing.get("created_at") or self._now(),
                "updated_at": self._now(),
            }
            self._write(data)
            return doc

    def remove_kb(self, kb_id: str) -> List[Dict]:
        with self._lock:
            data = self._read()
            removed = []
            keep = {}
            for doc_id, doc in data.get("documents", {}).items():
                if doc.get("kb_id") == kb_id:
                    removed.append(doc)
                else:
                    keep[doc_id] = doc
            data["documents"] = keep
            kb_existing = data.get("kbs", {}).get(kb_id, {})
            data.setdefault("kbs", {})[kb_id] = {
                "kb_id": kb_id,
                "created_at": kb_existing.get("created_at") or self._now(),
                "updated_at": self._now(),
            }
            self._write(data)
            return removed
