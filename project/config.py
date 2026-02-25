import os

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(__file__)
_DATA_ROOT = os.getenv("RAG_DATA_ROOT", _BASE_DIR)

MARKDOWN_DIR = os.getenv("RAG_MARKDOWN_DIR", os.path.join(_DATA_ROOT, "markdown_docs"))
PARENT_STORE_PATH = os.getenv("RAG_PARENT_STORE_PATH", os.path.join(_DATA_ROOT, "parent_store"))
QDRANT_DB_PATH = os.getenv("RAG_QDRANT_DB_PATH", os.path.join(_DATA_ROOT, "qdrant_db"))

# --- API Service Configuration ---
API_HOST = os.getenv("RAG_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("RAG_API_PORT", "8099"))
RAG_API_TOKEN = os.getenv("RAG_API_TOKEN", "change-me")
TASK_WORKERS = int(os.getenv("RAG_TASK_WORKERS", "1"))
DEFAULT_KB_ID = os.getenv("RAG_DEFAULT_KB_ID", "default")

KB_MARKDOWN_ROOT = os.getenv(
    "RAG_KB_MARKDOWN_ROOT", os.path.join(_DATA_ROOT, "markdown_docs_kb")
)
KB_PARENT_STORE_ROOT = os.getenv(
    "RAG_KB_PARENT_STORE_ROOT", os.path.join(_DATA_ROOT, "parent_store_kb")
)
KB_UPLOAD_ROOT = os.getenv(
    "RAG_KB_UPLOAD_ROOT", os.path.join(_DATA_ROOT, "uploads_kb")
)
TASK_STORE_PATH = os.getenv(
    "RAG_TASK_STORE_PATH", os.path.join(_DATA_ROOT, "task_store", "tasks.json")
)
DOCUMENT_INDEX_PATH = os.getenv(
    "RAG_DOCUMENT_INDEX_PATH", os.path.join(_DATA_ROOT, "document_index.json")
)

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Model Configuration ---
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
SPARSE_MODEL = "Qdrant/bm25"
LLM_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
LLM_TEMPERATURE = 0
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# --- Agent Configuration ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]
