import base64
import binascii
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import config
from db.document_index_manager import DocumentIndexManager
from db.parent_store_manager import ParentStoreManager
from db.vector_db_manager import VectorDbManager
from document_chunker import DocumentChuncker
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from services.task_manager import TaskManager
from utils import pdf_to_markdown


def _sanitize_kb_id(kb_id: Optional[str]) -> str:
    raw = (kb_id or config.DEFAULT_KB_ID).strip().lower()
    safe = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-")
    return safe or config.DEFAULT_KB_ID


class RagService:
    def __init__(self):
        self.vector_db = VectorDbManager()
        self.chunker = DocumentChuncker()
        self.index = DocumentIndexManager()
        self.tasks = TaskManager()
        self._qa_llm: Optional[ChatOllama] = None

    def _kb_collection_name(self, kb_id: str) -> str:
        return f"kb_{kb_id}_{config.CHILD_COLLECTION}"

    def _kb_markdown_dir(self, kb_id: str) -> Path:
        path = Path(config.KB_MARKDOWN_ROOT) / kb_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _kb_parent_store_dir(self, kb_id: str) -> Path:
        path = Path(config.KB_PARENT_STORE_ROOT) / kb_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _kb_upload_dir(self, kb_id: str) -> Path:
        path = Path(config.KB_UPLOAD_ROOT) / kb_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _parent_store(self, kb_id: str) -> ParentStoreManager:
        return ParentStoreManager(store_path=str(self._kb_parent_store_dir(kb_id)))

    def _next_available_markdown_path(self, kb_id: str, stem: str) -> Path:
        markdown_dir = self._kb_markdown_dir(kb_id)
        candidate = markdown_dir / f"{stem}.md"
        if not candidate.exists():
            return candidate
        i = 1
        while True:
            next_path = markdown_dir / f"{stem}-{i}.md"
            if not next_path.exists():
                return next_path
            i += 1

    def _ensure_collection(self, kb_id: str) -> str:
        collection_name = self._kb_collection_name(kb_id)
        self.vector_db.create_collection(collection_name)
        return collection_name

    def _ensure_kb(self, kb_id: str) -> str:
        kb = _sanitize_kb_id(kb_id)
        self._ensure_collection(kb)
        self._kb_markdown_dir(kb)
        self._kb_parent_store_dir(kb)
        self._kb_upload_dir(kb)
        self.index.ensure_kb(kb, touch_updated=False)
        return kb

    def create_kb(self, kb_id: str) -> Dict:
        kb = _sanitize_kb_id(kb_id)
        existing = {item.get("kb_id") for item in self.index.list_kbs()}
        self._ensure_kb(kb)

        summary = next((item for item in self.index.list_kbs() if item.get("kb_id") == kb), None) or {}
        return {
            "kb_id": kb,
            "created": kb not in existing,
            "document_count": int(summary.get("document_count") or 0),
            "last_updated_at": summary.get("last_updated_at"),
        }

    def list_kbs(self) -> List[Dict]:
        rows = self.index.list_kbs()
        if not any((item.get("kb_id") == config.DEFAULT_KB_ID) for item in rows):
            rows.insert(
                0,
                {
                    "kb_id": config.DEFAULT_KB_ID,
                    "document_count": 0,
                    "last_updated_at": None,
                    "created_at": None,
                },
            )
        return rows

    @staticmethod
    def _normalize_source_filters(source_names: Optional[Sequence[str]]) -> set:
        if not source_names:
            return set()
        return {name.strip().lower() for name in source_names if isinstance(name, str) and name.strip()}

    @staticmethod
    def _as_float_or_none(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _similarity_search(
        self,
        collection: Any,
        query: str,
        top_k: int,
        score_threshold: float,
    ) -> List[Tuple[Any, Optional[float]]]:
        results: List[Tuple[Any, Optional[float]]] = []

        try:
            pairs = collection.similarity_search_with_relevance_scores(
                query,
                k=top_k,
                score_threshold=score_threshold,
            )
            results = [(doc, self._as_float_or_none(score)) for doc, score in pairs]
        except Exception:
            try:
                pairs = collection.similarity_search_with_score(query, k=top_k)
                results = [(doc, self._as_float_or_none(score)) for doc, score in pairs]
            except Exception:
                try:
                    docs = collection.similarity_search(query, k=top_k, score_threshold=score_threshold)
                except TypeError:
                    docs = collection.similarity_search(query, k=top_k)
                results = [(doc, None) for doc in docs]

        if not results and score_threshold > 0:
            try:
                docs = collection.similarity_search(query, k=top_k)
                results = [(doc, None) for doc in docs]
            except Exception:
                return []

        return results

    @staticmethod
    def _short_snippet(text: str, limit: int = 320) -> str:
        compact = re.sub(r"\s+", " ", (text or "")).strip()
        return compact[:limit]

    @staticmethod
    def _build_ref_id(
        kb_id: str,
        document_id: Optional[str],
        parent_id: Optional[str],
        source_name: Optional[str],
        chunk_text: str,
    ) -> str:
        chunk_hash = hashlib.sha1(chunk_text.encode("utf-8", errors="ignore")).hexdigest()[:16]
        payload = {
            "v": 1,
            "kb_id": kb_id,
            "document_id": document_id or "",
            "parent_id": parent_id or "",
            "source_name": source_name or "",
            "chunk_hash": chunk_hash,
        }
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return f"r_{token}"

    @staticmethod
    def _decode_ref_id(ref_id: str) -> Optional[Dict]:
        if not isinstance(ref_id, str) or not ref_id.startswith("r_"):
            return None
        token = ref_id[2:].strip()
        if not token:
            return None
        padding = "=" * (-len(token) % 4)
        try:
            raw = base64.urlsafe_b64decode(token + padding)
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _extract_first_json_object(raw_text: str) -> Optional[Dict]:
        text = (raw_text or "").strip()
        if not text:
            return None

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else None
        except Exception:
            pass

        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        payload = json.loads(candidate)
                        return payload if isinstance(payload, dict) else None
                    except Exception:
                        return None
        return None

    def _qa_model(self) -> ChatOllama:
        if self._qa_llm is None:
            self._qa_llm = ChatOllama(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                base_url=config.OLLAMA_BASE_URL,
            )
        return self._qa_llm

    def _generate_answer_with_refs(
        self,
        question: str,
        hits: List[Dict],
        max_context_chunks: int,
    ) -> Tuple[str, List[str]]:
        if not hits:
            return "I could not find relevant information in the selected knowledge base.", []

        context_hits = hits[: max(1, max_context_chunks)]
        available_refs = [hit.get("ref_id") for hit in context_hits if hit.get("ref_id")]
        available_ref_set = set(available_refs)

        context_blocks = []
        for hit in context_hits:
            ref_id = hit.get("ref_id")
            chunk_text = hit.get("chunk_text", "")
            context_blocks.append(f"[ref_id={ref_id}]\n{chunk_text}")

        system_prompt = (
            "You are a retrieval QA assistant. Use ONLY the provided context blocks. "
            "Return STRICT JSON with keys: answer (string), used_refs (array of ref_id strings). "
            "Do not include ref_ids that are not present in the context."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Allowed ref_ids: {available_refs}\n\n"
            "Context blocks:\n"
            + "\n\n".join(context_blocks)
        )

        try:
            model = self._qa_model()
            response = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            raw_text = str(response.content or "").strip()
        except Exception:
            return "I could not generate an answer due to an LLM runtime error.", []

        parsed = self._extract_first_json_object(raw_text)
        answer = raw_text
        raw_used_refs: Any = []
        if parsed:
            parsed_answer = parsed.get("answer")
            if isinstance(parsed_answer, str) and parsed_answer.strip():
                answer = parsed_answer.strip()
            raw_used_refs = parsed.get("used_refs") or []

        if isinstance(raw_used_refs, str):
            raw_used_refs = [raw_used_refs]
        if not isinstance(raw_used_refs, list):
            raw_used_refs = []

        used_refs: List[str] = []
        for value in raw_used_refs:
            if not isinstance(value, str):
                continue
            ref_id = value.strip()
            if ref_id and ref_id in available_ref_set and ref_id not in used_refs:
                used_refs.append(ref_id)

        if not used_refs:
            for match in re.findall(r"r_[A-Za-z0-9_-]+", answer):
                if match in available_ref_set and match not in used_refs:
                    used_refs.append(match)

        return answer, used_refs

    def submit_ingest_task(
        self,
        kb_id: str,
        source_type: str,
        source: str,
        source_name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        dedupe_key: Optional[str] = None,
    ) -> Dict:
        kb = _sanitize_kb_id(kb_id)
        payload = {
            "kb_id": kb,
            "source_type": source_type,
            "source": source,
            "source_name": source_name,
            "metadata": metadata or {},
            "dedupe_key": dedupe_key,
        }
        return self.tasks.submit("ingest_document", payload, self._ingest_worker)

    def _ingest_worker(self, payload: Dict) -> Dict:
        kb_id = self._ensure_kb(payload["kb_id"])
        source_type = payload["source_type"]
        source = payload["source"]
        source_name = payload.get("source_name")
        metadata = payload.get("metadata") or {}
        dedupe_key = payload.get("dedupe_key")

        if dedupe_key:
            existing = self.index.find_by_dedupe_key(kb_id, dedupe_key)
            if existing:
                return {
                    "status": "skipped",
                    "reason": "dedupe_key_exists",
                    "document_id": existing["document_id"],
                }

        collection = self.vector_db.get_collection(self._kb_collection_name(kb_id))
        parent_store = self._parent_store(kb_id)

        source_name, markdown_path = self._materialize_markdown(
            kb_id, source_type, source, source_name=source_name
        )
        document_id = str(uuid.uuid4())

        parent_pairs, child_chunks = self.chunker.create_chunks_single(markdown_path)
        if not child_chunks:
            raise ValueError("No child chunks were generated from document")

        parent_ids: List[str] = []
        for parent_id, parent_doc in parent_pairs:
            parent_doc.metadata.update(
                {
                    "document_id": document_id,
                    "kb_id": kb_id,
                    "source_name": source_name,
                }
            )
            parent_ids.append(parent_id)

        for child_doc in child_chunks:
            child_doc.metadata.update(
                {
                    "document_id": document_id,
                    "kb_id": kb_id,
                    "source_name": source_name,
                }
            )

        collection.add_documents(child_chunks)
        parent_store.save_many(parent_pairs)

        self.index.upsert_document(
            kb_id=kb_id,
            document_id=document_id,
            source_name=source_name,
            markdown_path=str(markdown_path),
            parent_ids=parent_ids,
            child_count=len(child_chunks),
            metadata=metadata,
            dedupe_key=dedupe_key,
        )

        return {
            "status": "completed",
            "kb_id": kb_id,
            "document_id": document_id,
            "source_name": source_name,
            "markdown_path": str(markdown_path),
            "parent_count": len(parent_ids),
            "child_count": len(child_chunks),
        }

    def _materialize_markdown(
        self,
        kb_id: str,
        source_type: str,
        source: str,
        source_name: Optional[str] = None,
    ):
        if source_type not in {"file_path", "text", "base64_file"}:
            raise ValueError("source_type must be one of: file_path, text, base64_file")

        if source_type == "text":
            stem = f"text-{uuid.uuid4().hex[:8]}"
            md_path = self._next_available_markdown_path(kb_id, stem)
            md_path.write_text(source, encoding="utf-8")
            return f"{stem}.md", md_path

        if source_type == "base64_file":
            normalized_source_name = Path(source_name or "").name.strip()
            if not normalized_source_name:
                raise ValueError("source_name is required for base64_file source_type")

            suffix = Path(normalized_source_name).suffix.lower()
            if suffix not in {".pdf", ".md"}:
                raise ValueError("Only .pdf and .md files are supported for base64_file")

            try:
                raw_bytes = base64.b64decode(source, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("source must be valid base64-encoded content") from exc

            md_target_path = self._next_available_markdown_path(
                kb_id, Path(normalized_source_name).stem
            )
            if suffix == ".md":
                try:
                    md_target_path.write_text(raw_bytes.decode("utf-8"), encoding="utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("Markdown file must be UTF-8 encoded") from exc
                return normalized_source_name, md_target_path

            upload_name = f"{md_target_path.stem}.pdf"
            upload_path = self._kb_upload_dir(kb_id) / upload_name
            upload_path.write_bytes(raw_bytes)
            try:
                pdf_to_markdown(
                    str(upload_path),
                    str(self._kb_markdown_dir(kb_id)),
                    output_name=md_target_path.stem,
                )
            finally:
                if upload_path.exists():
                    upload_path.unlink()
            return normalized_source_name, md_target_path

        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source file does not exist: {source}")

        suffix = source_path.suffix.lower()
        if suffix not in {".pdf", ".md"}:
            raise ValueError("Only .pdf and .md files are supported")

        md_target_path = self._next_available_markdown_path(kb_id, source_path.stem)
        if suffix == ".md":
            shutil.copy2(source_path, md_target_path)
        else:
            pdf_to_markdown(
                str(source_path),
                str(self._kb_markdown_dir(kb_id)),
                output_name=md_target_path.stem,
            )
        return source_path.name, md_target_path

    def get_task(self, task_id: str) -> Dict:
        task = self.tasks.get(task_id)
        if not task:
            return {}
        return task

    def list_documents(self, kb_id: str) -> List[Dict]:
        kb = _sanitize_kb_id(kb_id)
        return self.index.list_documents(kb)

    def retrieve(
        self,
        kb_id: str,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        source_names: Optional[Sequence[str]] = None,
    ) -> Dict:
        kb = self._ensure_kb(kb_id)
        collection = self.vector_db.get_collection(self._kb_collection_name(kb))
        normalized_filters = self._normalize_source_filters(source_names)
        candidate_k = min(200, top_k * 5 if normalized_filters else top_k)
        candidate_pairs = self._similarity_search(collection, query, top_k=max(1, candidate_k), score_threshold=score_threshold)

        filtered_out = 0
        hits = []
        for doc, score in candidate_pairs:
            metadata = getattr(doc, "metadata", {}) or {}
            source_name = (metadata.get("source_name") or metadata.get("source") or "").strip()
            if normalized_filters and source_name.lower() not in normalized_filters:
                filtered_out += 1
                continue

            document_id = metadata.get("document_id")
            parent_id = metadata.get("parent_id")
            chunk_text = getattr(doc, "page_content", "") or ""
            ref_id = self._build_ref_id(
                kb_id=kb,
                document_id=document_id,
                parent_id=parent_id,
                source_name=source_name,
                chunk_text=chunk_text,
            )

            hits.append(
                {
                    "ref_id": ref_id,
                    "chunk_text": chunk_text,
                    "score": score,
                }
            )
            if len(hits) >= top_k:
                break

        return {
            "kb_id": kb,
            "query": query,
            "top_k": top_k,
            "hits": hits,
            "debug": {
                "candidate_count": len(candidate_pairs),
                "filtered_out_by_source": filtered_out,
                "returned_count": len(hits),
                "score_threshold": score_threshold,
                "source_filter": sorted(normalized_filters),
            },
        }

    def resolve_refs(self, kb_id: str, ref_ids: Sequence[str]) -> Dict:
        kb = _sanitize_kb_id(kb_id)
        citations = []
        unresolved = []
        seen = set()

        for raw_ref in ref_ids or []:
            if not isinstance(raw_ref, str):
                continue
            ref_id = raw_ref.strip()
            if not ref_id or ref_id in seen:
                continue
            seen.add(ref_id)

            decoded = self._decode_ref_id(ref_id)
            if not decoded:
                unresolved.append({"ref_id": ref_id, "reason": "invalid_ref_id"})
                continue

            ref_kb = str(decoded.get("kb_id") or "").strip()
            if ref_kb and ref_kb != kb:
                unresolved.append({"ref_id": ref_id, "reason": "kb_mismatch"})
                continue

            document_id = str(decoded.get("document_id") or "").strip() or None
            parent_id = str(decoded.get("parent_id") or "").strip() or None
            source_name = str(decoded.get("source_name") or "").strip() or None
            page_hint = None
            snippet = ""

            if parent_id:
                try:
                    parent_doc = self._parent_store(kb).load_content(parent_id)
                    parent_meta = parent_doc.get("metadata", {}) or {}
                    if not source_name:
                        source_name = parent_meta.get("source_name") or parent_meta.get("source")
                    page_hint = (
                        parent_meta.get("page_hint")
                        or parent_meta.get("page")
                        or parent_meta.get("page_number")
                    )
                    snippet = self._short_snippet(parent_doc.get("content", ""))
                except Exception:
                    pass

            if document_id and not source_name:
                doc_record = self.index.get_document(kb, document_id)
                if doc_record:
                    source_name = doc_record.get("source_name")

            citation = {
                "ref_id": ref_id,
                "source_name": source_name,
                "document_id": document_id,
                "parent_id": parent_id,
                "page_hint": page_hint,
                "snippet": snippet,
            }
            citations.append(citation)

        return {
            "kb_id": kb,
            "citations": citations,
            "unresolved": unresolved,
            "count": len(citations),
        }

    def ask(
        self,
        kb_id: str,
        question: str,
        top_k: int = 5,
        thread_id: Optional[str] = None,
        max_context_parents: int = 5,
        source_names: Optional[Sequence[str]] = None,
        include_debug: bool = True,
    ) -> Dict:
        kb = _sanitize_kb_id(kb_id)
        active_thread_id = thread_id or str(uuid.uuid4())

        retrieval = self.retrieve(
            kb_id=kb,
            query=question,
            top_k=top_k,
            source_names=source_names,
        )
        hits = retrieval.get("hits") or []
        context_limit = min(len(hits), max(1, int(max_context_parents)))
        answer, used_refs = self._generate_answer_with_refs(
            question=question,
            hits=hits,
            max_context_chunks=context_limit,
        )
        resolved = self.resolve_refs(kb, used_refs)

        result = {
            "kb_id": kb,
            "thread_id": active_thread_id,
            "answer": answer,
            "used_refs": used_refs,
            "citations": resolved.get("citations", []),
        }
        if include_debug:
            result["debug"] = {
                "retrieval_stats": retrieval.get("debug", {}),
                "retrieval_count": len(hits),
                "context_chunk_count": context_limit,
                "unresolved_refs": resolved.get("unresolved", []),
            }
        return result

    def delete_document(self, kb_id: str, document_id: str) -> Dict:
        kb = self._ensure_kb(kb_id)
        record = self.index.get_document(kb, document_id)
        if not record:
            return {"deleted": False, "reason": "document_not_found"}

        collection_name = self._kb_collection_name(kb)
        parent_ids = record.get("parent_ids", [])
        parent_store = self._parent_store(kb)

        if parent_ids:
            self.vector_db.delete_by_parent_ids(collection_name, parent_ids)
        else:
            self.vector_db.delete_by_document_id(collection_name, document_id)
        removed_parents = parent_store.delete_many(parent_ids)

        markdown_path = record.get("markdown_path")
        markdown_removed = False
        if markdown_path and Path(markdown_path).exists():
            Path(markdown_path).unlink()
            markdown_removed = True

        self.index.remove_document(kb, document_id)
        return {
            "deleted": True,
            "kb_id": kb,
            "document_id": document_id,
            "removed_parents": removed_parents,
            "removed_vectors_for_parent_ids": len(parent_ids),
            "removed_markdown": markdown_removed,
        }

    def submit_reindex_task(self, kb_id: str, document_ids: Optional[List[str]] = None) -> Dict:
        kb = self._ensure_kb(kb_id)
        payload = {"kb_id": kb, "document_ids": document_ids or []}
        return self.tasks.submit("reindex", payload, self._reindex_worker)

    def _reindex_worker(self, payload: Dict) -> Dict:
        kb = self._ensure_kb(payload["kb_id"])
        target_ids = set(payload.get("document_ids") or [])
        docs = self.index.list_documents(kb)
        if target_ids:
            docs = [d for d in docs if d.get("document_id") in target_ids]

        collection_name = self._ensure_collection(kb)
        collection = self.vector_db.get_collection(collection_name)
        parent_store = self._parent_store(kb)

        updated = 0
        for doc in docs:
            document_id = doc["document_id"]
            old_parent_ids = doc.get("parent_ids", [])
            if old_parent_ids:
                self.vector_db.delete_by_parent_ids(collection_name, old_parent_ids)
                parent_store.delete_many(old_parent_ids)

            md_path = doc.get("markdown_path")
            if not md_path or not Path(md_path).exists():
                continue

            parent_pairs, child_chunks = self.chunker.create_chunks_single(md_path)
            parent_ids = []
            for parent_id, parent_doc in parent_pairs:
                parent_doc.metadata.update(
                    {
                        "document_id": document_id,
                        "kb_id": kb,
                        "source_name": doc.get("source_name"),
                    }
                )
                parent_ids.append(parent_id)

            for child_doc in child_chunks:
                child_doc.metadata.update(
                    {
                        "document_id": document_id,
                        "kb_id": kb,
                        "source_name": doc.get("source_name"),
                    }
                )

            collection.add_documents(child_chunks)
            parent_store.save_many(parent_pairs)

            self.index.upsert_document(
                kb_id=kb,
                document_id=document_id,
                source_name=doc.get("source_name"),
                markdown_path=md_path,
                parent_ids=parent_ids,
                child_count=len(child_chunks),
                metadata=doc.get("metadata") or {},
                dedupe_key=doc.get("dedupe_key"),
            )
            updated += 1

        return {"kb_id": kb, "reindexed_documents": updated}

    def clear_kb(self, kb_id: str) -> Dict:
        kb = self._ensure_kb(kb_id)
        collection_name = self._kb_collection_name(kb)

        self.vector_db.delete_collection(collection_name)
        self.vector_db.create_collection(collection_name)

        parent_dir = self._kb_parent_store_dir(kb)
        markdown_dir = self._kb_markdown_dir(kb)
        removed_parent_files = 0
        removed_markdown_files = 0

        if parent_dir.exists():
            removed_parent_files = len(list(parent_dir.glob("*.json")))
            shutil.rmtree(parent_dir)
        if markdown_dir.exists():
            removed_markdown_files = len(list(markdown_dir.glob("*.md")))
            shutil.rmtree(markdown_dir)

        parent_dir.mkdir(parents=True, exist_ok=True)
        markdown_dir.mkdir(parents=True, exist_ok=True)

        removed_docs = self.index.remove_kb(kb)
        self.index.ensure_kb(kb)

        return {
            "kb_id": kb,
            "cleared": True,
            "removed_documents": len(removed_docs),
            "removed_parent_files": removed_parent_files,
            "removed_markdown_files": removed_markdown_files,
        }
