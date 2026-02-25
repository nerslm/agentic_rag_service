# Agentic RAG Service

一个面向 Agent 生态的 RAG 系统项目。我把早期的 Agentic RAG 原型（多步检索推理）工程化为可部署的 HTTP 服务，并完成了与 OpenClaw 的工具化集成。

这个仓库用于展示我在以下方面的完整能力：
- RAG 架构设计与迭代
- 检索质量与引用可解释性
- API 工程化与异步任务化
- Agent Tool 集成与落地部署

## 项目定位

目标不是“能跑的 Demo”，而是一个可持续演进的 Agent-ready RAG 基础设施：
- 多知识库隔离（`kb_id`）
- 可控检索上下文（减少 metadata 污染）
- 可追溯引用（citation 延迟绑定）
- 稳定 API 契约（方便 OpenClaw/其他 agent 接入）

## 我做了什么（核心工作）

1. Agentic RAG 原型能力（前期）
- 完成文档切分、向量检索、父子块回溯等基础能力。
- 验证“检索-推理-回答”闭环，并沉淀为可复用流程。

2. 服务化改造（当前仓库主线）
- 将原型改造为 FastAPI 服务，统一鉴权与错误码。
- 引入异步任务（入库/重建索引），通过 `task_id` 追踪状态。
- 完成多 KB 数据隔离与文献级过滤（`source_names`）。

3. 引用体系重构（关键设计）
- `retrieve` 只返回 `ref_id + chunk_text (+score)`，不把大段 metadata 注入 LLM。
- `ask` 输出 `answer + used_refs`。
- `resolve_refs` 单独做 citation 解析，返回 `source_name/document_id/page_hint/snippet`。

4. OpenClaw 工具集成
- 对接 `rag_*` 工具调用契约，支持建库、导入、检索、问答、删除、重建。
- 适配 Agent 调用场景，区分读操作和写操作权限边界。

## 架构概览

```text
OpenClaw UI / Agent Tools
          |
          v
  FastAPI API Layer (api_server.py)
          |
          v
        RagService
   /         |            \
Qdrant   ParentStore   DocumentIndex
```

核心问答链路：

1. `POST /v1/kb/{kb_id}/retrieve`
2. 构造最小上下文（仅 `ref_id + chunk_text`）
3. LLM 生成 `answer + used_refs`
4. `POST /v1/kb/{kb_id}/resolve_refs`
5. 返回 `answer + citations (+debug)`

## 关键设计决策

### 1) Two-stage RAG + Delayed Citation Binding
- 好处：降低上下文噪声，避免 metadata 干扰推理。
- 好处：引用展示与问答解耦，UI 可独立优化 citation 呈现。

### 2) 多知识库隔离
- 每个接口显式带 `kb_id`。
- 默认单库检索，避免跨领域语义污染。

### 3) 异步任务化
- 入库/重建索引作为后台任务执行，前台仅拿 `task_id`。
- 任务状态：`queued/running/completed/failed/skipped`。

### 4) 面向工具调用的 API 契约
- 请求/响应结构稳定，适配 Tool Calling。
- 错误返回统一结构，便于 agent 做重试与兜底。

## 技术栈

- Python 3.13
- FastAPI + Uvicorn
- Qdrant（向量检索）
- Ollama（本地 LLM）
- LangChain（模型调用封装）
- Docker（单容器 API+Ollama 部署）

## 主要接口

- `GET /healthz`
- `GET /v1/kb`
- `POST /v1/kb/{kb_id}`
- `POST /v1/kb/{kb_id}/documents`
- `GET /v1/tasks/{task_id}`
- `GET /v1/kb/{kb_id}/documents`
- `DELETE /v1/kb/{kb_id}/documents/{document_id}`
- `POST /v1/kb/{kb_id}/retrieve`
- `POST /v1/kb/{kb_id}/resolve_refs`
- `POST /v1/kb/{kb_id}/ask`
- `POST /v1/kb/{kb_id}/reindex`
- `POST /v1/kb/{kb_id}/clear`

详细请求示例见：[project/API_README.md](/home/test/workspace/agentic_rag_service/project/API_README.md)

## 快速开始

### 方式 A：本地启动

```bash
cd project
export RAG_API_TOKEN="replace-with-strong-token"
python3 api_server.py
```

默认：`http://127.0.0.1:8099`

### 方式 B：Docker（推荐演示）

```bash
cd /home/test/workspace/agentic_rag_service
docker build -f Dockerfile.api-ollama -t rag-api-ollama:local .

docker run -d --name rag-api-ollama \
  -p 8099:8099 \
  -e RAG_API_TOKEN="rag-dev-token-123" \
  -e RAG_DATA_ROOT="/data" \
  -v /home/test/workspace/agentic_rag_service/.data:/data \
  rag-api-ollama:local
```

健康检查：

```bash
curl -sS http://127.0.0.1:8099/healthz \
  -H "Authorization: Bearer rag-dev-token-123"
```

## 环境变量

- `RAG_API_TOKEN`：API 鉴权 token
- `RAG_API_HOST` / `RAG_API_PORT`：服务监听地址
- `RAG_DATA_ROOT`：数据根目录（索引、任务、文档）
- `RAG_DEFAULT_KB_ID`：默认知识库
- `OLLAMA_BASE_URL`：LLM 服务地址
- `RAG_OLLAMA_MODEL`：启动时模型名
- `OLLAMA_PULL_ON_START`：是否启动时拉模型（`1/0`）

## 与 OpenClaw 集成（实践）

我在 OpenClaw 侧按工具插件方式接入该服务，常用工具包括：
- 读能力：`rag_retrieve`、`rag_ask`、`rag_list_documents`、`rag_list_kb`
- 写能力：`rag_create_kb`、`rag_ingest`、`rag_delete_document`、`rag_reindex`
- 任务与引用：`rag_task_status`、`rag_resolve_refs`

实践经验：
- 将读写工具做权限分离，减少误操作。
- 将 citation 解析交给 `resolve_refs`，避免 LLM 幻觉引用。

## 仓库结构（精简后）

```text
project/
  api_server.py            # FastAPI 入口
  api/schemas.py           # 请求模型
  services/rag_service.py  # 核心业务逻辑
  services/task_manager.py # 异步任务管理
  db/                      # 向量库/索引/父块存储
  document_chunker.py      # 文档切分
  docker/start_api_ollama.sh
```

## 后续规划

- 引入 rerank（cross-encoder）提升高相似噪声场景精度
- 增加离线评测（Recall@K、Citation Hit Rate）
- 增加流式响应与任务事件推送（WebSocket）
- 增加多租户与访问策略控制

## License

MIT
