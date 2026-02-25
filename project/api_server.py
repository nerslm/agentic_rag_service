import os
import sys
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))

import config
from api.schemas import (
    AskRequest,
    ClearRequest,
    IngestRequest,
    ReindexRequest,
    ResolveRefsRequest,
    RetrieveRequest,
)
from services.rag_service import RagService


app = FastAPI(title="Agentic RAG API", version="1.0.0")
auth_scheme = HTTPBearer(auto_error=False)
rag_service = RagService()


def _ok(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "data": data}


def _err(code: str, message: str, details: Any = None, http_status: int = status.HTTP_400_BAD_REQUEST):
    payload = {"ok": False, "error": {"code": code, "message": message}}
    if details is not None:
        payload["error"]["details"] = details
    raise HTTPException(status_code=http_status, detail=payload)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials is None or credentials.scheme.lower() != "bearer":
        _err(
            "UNAUTHORIZED",
            "Missing bearer token",
            http_status=status.HTTP_401_UNAUTHORIZED,
        )
    if credentials.credentials != config.RAG_API_TOKEN:
        _err(
            "UNAUTHORIZED",
            "Invalid token",
            http_status=status.HTTP_401_UNAUTHORIZED,
        )
    return True


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    if isinstance(exc.detail, dict) and exc.detail.get("ok") is False:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": {"code": "HTTP_ERROR", "message": str(exc.detail)}},
    )


@app.get("/healthz")
def healthz(_: bool = Depends(verify_token)):
    return _ok({"status": "ok", "version": "1.0.0"})


@app.post("/v1/kb/{kb_id}/documents")
def create_document(kb_id: str, request: IngestRequest, _: bool = Depends(verify_token)):
    try:
        task = rag_service.submit_ingest_task(
            kb_id=kb_id,
            source_type=request.source_type,
            source=request.source,
            source_name=request.source_name,
            metadata=request.metadata,
            dedupe_key=request.dedupe_key,
        )
        return _ok(
            {
                "task_id": task["task_id"],
                "status": task["status"],
                "task_type": task["task_type"],
            }
        )
    except Exception as e:
        _err("INGEST_SUBMIT_FAILED", str(e), http_status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.get("/v1/tasks/{task_id}")
def get_task(task_id: str, _: bool = Depends(verify_token)):
    task = rag_service.get_task(task_id)
    if not task:
        _err("TASK_NOT_FOUND", "Task not found", http_status=status.HTTP_404_NOT_FOUND)
    return _ok(task)


@app.get("/v1/kb")
def list_kbs(_: bool = Depends(verify_token)):
    kbs = rag_service.list_kbs()
    return _ok({"kbs": kbs, "count": len(kbs)})


@app.post("/v1/kb/{kb_id}")
def create_kb(kb_id: str, _: bool = Depends(verify_token)):
    try:
        result = rag_service.create_kb(kb_id)
        return _ok(result)
    except Exception as e:
        _err("KB_CREATE_FAILED", str(e), http_status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.get("/v1/kb/{kb_id}/documents")
def list_documents(kb_id: str, _: bool = Depends(verify_token)):
    docs = rag_service.list_documents(kb_id)
    return _ok({"kb_id": kb_id, "documents": docs, "count": len(docs)})


@app.delete("/v1/kb/{kb_id}/documents/{document_id}")
def delete_document(kb_id: str, document_id: str, _: bool = Depends(verify_token)):
    try:
        result = rag_service.delete_document(kb_id, document_id)
        if not result.get("deleted"):
            _err("DOCUMENT_NOT_FOUND", "Document not found", http_status=status.HTTP_404_NOT_FOUND)
        return _ok(result)
    except HTTPException:
        raise
    except Exception as e:
        _err("DELETE_FAILED", str(e), http_status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/v1/kb/{kb_id}/reindex")
def reindex(kb_id: str, request: ReindexRequest, _: bool = Depends(verify_token)):
    try:
        task = rag_service.submit_reindex_task(kb_id, request.document_ids)
        return _ok({"task_id": task["task_id"], "status": task["status"], "task_type": task["task_type"]})
    except Exception as e:
        _err("REINDEX_SUBMIT_FAILED", str(e), http_status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/v1/kb/{kb_id}/retrieve")
def retrieve(kb_id: str, request: RetrieveRequest, _: bool = Depends(verify_token)):
    try:
        result = rag_service.retrieve(
            kb_id=kb_id,
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            source_names=request.source_names,
        )
        return _ok(result)
    except Exception as e:
        _err("RETRIEVE_FAILED", str(e), http_status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/v1/kb/{kb_id}/resolve_refs")
def resolve_refs(kb_id: str, request: ResolveRefsRequest, _: bool = Depends(verify_token)):
    try:
        result = rag_service.resolve_refs(
            kb_id=kb_id,
            ref_ids=request.ref_ids,
        )
        return _ok(result)
    except Exception as e:
        _err("RESOLVE_REFS_FAILED", str(e), http_status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/v1/kb/{kb_id}/ask")
def ask(kb_id: str, request: AskRequest, _: bool = Depends(verify_token)):
    try:
        result = rag_service.ask(
            kb_id=kb_id,
            question=request.question,
            top_k=request.top_k,
            thread_id=request.thread_id,
            max_context_parents=request.max_context_parents,
            source_names=request.source_names,
            include_debug=request.include_debug,
        )
        return _ok(result)
    except Exception as e:
        _err("ASK_FAILED", str(e), http_status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/v1/kb/{kb_id}/clear")
def clear_kb(kb_id: str, request: ClearRequest, _: bool = Depends(verify_token)):
    if not request.confirm:
        _err("CONFIRM_REQUIRED", "Set confirm=true to clear knowledge base")
    try:
        result = rag_service.clear_kb(kb_id)
        return _ok(result)
    except Exception as e:
        _err("CLEAR_FAILED", str(e), http_status=status.HTTP_500_INTERNAL_SERVER_ERROR)


if __name__ == "__main__":
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
