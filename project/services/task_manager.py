import json
import queue
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

import config


TaskHandler = Callable[[Dict[str, Any]], Dict[str, Any]]


class TaskManager:
    def __init__(self, workers: int = config.TASK_WORKERS, store_path: str = config.TASK_STORE_PATH):
        self._workers = max(1, int(workers))
        self._store_path = Path(store_path)
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._queue: "queue.Queue[tuple[str, Dict[str, Any], TaskHandler]]" = queue.Queue()
        self._lock = threading.Lock()
        self._load_store()
        self._start_workers()

    def _load_store(self) -> None:
        if not self._store_path.exists():
            return
        try:
            raw = self._store_path.read_text(encoding="utf-8")
            self._tasks = json.loads(raw) if raw else {}
        except Exception:
            self._tasks = {}

    def _persist(self) -> None:
        self._store_path.write_text(
            json.dumps(self._tasks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _start_workers(self) -> None:
        for idx in range(self._workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"rag-task-worker-{idx}",
                daemon=True,
            )
            thread.start()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _worker_loop(self) -> None:
        while True:
            task_id, payload, handler = self._queue.get()
            with self._lock:
                task = self._tasks.get(task_id, {})
                task["status"] = "running"
                task["started_at"] = self._now()
                self._tasks[task_id] = task
                self._persist()

            try:
                result = handler(payload) or {}
                result_status = "completed"
                if isinstance(result, dict):
                    raw_status = str(result.get("status") or "").strip().lower()
                    if raw_status in {"completed", "failed", "skipped"}:
                        result_status = raw_status
                with self._lock:
                    task = self._tasks.get(task_id, {})
                    task["status"] = result_status
                    task["completed_at"] = self._now()
                    task["result"] = result
                    self._tasks[task_id] = task
                    self._persist()
            except Exception as e:
                with self._lock:
                    task = self._tasks.get(task_id, {})
                    task["status"] = "failed"
                    task["completed_at"] = self._now()
                    task["error"] = {
                        "message": str(e),
                        "traceback": traceback.format_exc(limit=10),
                    }
                    self._tasks[task_id] = task
                    self._persist()
            finally:
                self._queue.task_done()

    def submit(self, task_type: str, payload: Dict[str, Any], handler: TaskHandler) -> Dict[str, Any]:
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "task_type": task_type,
            "status": "queued",
            "created_at": self._now(),
            "payload": payload,
        }
        with self._lock:
            self._tasks[task_id] = task
            self._persist()
        self._queue.put((task_id, payload, handler))
        return task

    def get(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._tasks.get(task_id, {})
