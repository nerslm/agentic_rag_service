# Agentic RAG HTTP API

This project now includes a FastAPI service for external calls.

## Start

```bash
cd project
export RAG_API_TOKEN="replace-with-strong-token"
python3 api_server.py
```

Default address: `http://127.0.0.1:8099`

## Start With Single Docker Container (includes Ollama)

Build image:

```bash
cd /home/test/workspace/agentic-rag-for-dummies
docker build -f Dockerfile.api-ollama -t agentic-rag-api:ollama .
```

Run container:

```bash
docker run -d --name agentic-rag-api \
  -p 8099:8099 \
  -e RAG_API_TOKEN="rag-dev-token-123" \
  -e RAG_OLLAMA_MODEL="qwen3:4b-instruct-2507-q4_K_M" \
  -e RAG_DATA_ROOT="/data" \
  -v rag_ollama_data:/root/.ollama \
  -v rag_data:/data \
  agentic-rag-api:ollama
```

Notes:
- First startup pulls the Ollama model and can take several minutes.
- If you do not want pull-on-start, set `-e OLLAMA_PULL_ON_START=0`.

Health check:

```bash
curl -sS http://127.0.0.1:8099/healthz \
  -H "Authorization: Bearer rag-dev-token-123"
```

## Authentication

All endpoints require:

```http
Authorization: Bearer <RAG_API_TOKEN>
```

## Core endpoints

- `GET /healthz`
- `GET /v1/kb`
- `POST /v1/kb/{kb_id}`
- `POST /v1/kb/{kb_id}/documents` (async ingest)
- `GET /v1/tasks/{task_id}`
- `GET /v1/kb/{kb_id}/documents`
- `DELETE /v1/kb/{kb_id}/documents/{document_id}`
- `POST /v1/kb/{kb_id}/retrieve`
- `POST /v1/kb/{kb_id}/resolve_refs`
- `POST /v1/kb/{kb_id}/ask`
- `POST /v1/kb/{kb_id}/reindex`
- `POST /v1/kb/{kb_id}/clear`

## Example: ingest + poll

```bash
curl -sS http://127.0.0.1:8099/v1/kb/default/documents \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"source_type":"file_path","source":"/abs/path/paper.pdf"}'
```

If API cannot access your local file path, upload file bytes:

```bash
FILE_B64="$(base64 -w 0 /home/test/workspace/EB_JEPA.pdf)"
curl -sS http://127.0.0.1:8099/v1/kb/default/documents \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"source_type\":\"base64_file\",\"source_name\":\"EB_JEPA.pdf\",\"source\":\"${FILE_B64}\"}"
```

```bash
curl -sS http://127.0.0.1:8099/v1/tasks/<task_id> \
  -H "Authorization: Bearer $RAG_API_TOKEN"
```

## Example: ask

```bash
curl -sS http://127.0.0.1:8099/v1/kb/default/ask \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize this paper"}'
```

## Example: retrieve + resolve refs

```bash
curl -sS http://127.0.0.1:8099/v1/kb/default/retrieve \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is JEPA?","top_k":5}'
```

Then call `resolve_refs` with returned `ref_id` values:

```bash
curl -sS http://127.0.0.1:8099/v1/kb/default/resolve_refs \
  -H "Authorization: Bearer $RAG_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"ref_ids":["r_xxx","r_yyy"]}'
```
