from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_type: Literal["file_path", "text", "base64_file"] = "file_path"
    source: str = Field(min_length=1)
    source_name: Optional[str] = None
    metadata: Optional[Dict] = None
    dedupe_key: Optional[str] = None


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    source_names: Optional[List[str]] = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    thread_id: Optional[str] = None
    max_context_parents: int = Field(default=5, ge=1, le=20)
    source_names: Optional[List[str]] = None
    include_debug: bool = True


class ReindexRequest(BaseModel):
    document_ids: Optional[List[str]] = None


class ResolveRefsRequest(BaseModel):
    ref_ids: List[str] = Field(default_factory=list)


class ClearRequest(BaseModel):
    confirm: bool = False
