# Grok2API-tqzhr 功能同步记录

## 2026-03-21

## 2026-03-11 — 从 grok2api-chenyme 同步新增功能

### A1: Chat 新增 reasoning_effort/temperature/top_p 参数
- **问题**: tqzhr 缺少 chenyme 新增的 reasoning_effort、temperature、top_p 参数
- **方案**: 在 ChatCompletionRequest 中添加参数，reasoning_effort 映射到 thinking 逻辑
- **修改文件**: `app/api/v1/chat.py`

### A4: 新增模型 grok-imagine-1.0-fast + is_image_edit 字段
- **问题**: tqzhr 缺少 grok-imagine-1.0-fast 模型和 is_image_edit 字段
- **方案**: 在 ModelInfo 中添加 is_image_edit 字段，在模型列表中添加 grok-imagine-1.0-fast
- **修改文件**: `app/services/grok/model.py`

### A2: Chat Tool Calling 支持
- **问题**: tqzhr 不支持 OpenAI 兼容的 tool calling
- **方案**: 移植 chenyme 的 tool_call.py 工具函数，集成到 chat 端点（prompt 注入 + 响应解析）
- **修改文件**: 
  - `app/services/grok/tool_call.py`（新建）
  - `app/api/v1/chat.py`

### A3: Responses API (/v1/responses) 路由和服务
- **问题**: tqzhr 缺少 OpenAI Responses API 兼容端点
- **方案**: 从 chenyme 移植 responses 服务，适配 tqzhr 的 ChatService 架构（通过 tool_call.py 处理工具调用）
- **修改文件**:
  - `app/services/grok/responses.py`（新建）
  - `app/api/v1/response.py`（新建）
  - `main.py`（注册路由）

### A5: Chat 内联 image_config 支持
- **问题**: chat 端点不支持图片模型路由（grok-imagine-1.0-fast 等）
- **方案**: 在 chat 端点检测 is_image 模型，提取 prompt，路由到图片生成基础设施，返回 chat completion 格式
- **修改文件**: `app/api/v1/chat.py`

### B6: 视频 API /v1/videos
- **问题**: tqzhr 的 video.py 是空占位文件
- **方案**: 实现 /v1/videos 端点，支持 JSON 和 multipart/form-data，使用 tqzhr 已有的 VideoService
- **修改文件**:
  - `app/api/v1/video.py`（重写）
  - `main.py`（注册路由）

### B7: Batch 批处理工具移植
- **问题**: tqzhr 缺少批处理工具
- **方案**: 移植 chenyme 的 batch.py（run_batch + BatchTask SSE 任务管理器）
- **修改文件**: `app/core/batch.py`（新建）

### 修复: Workers 模型目录同步
- **问题**: CI 检查 `check_model_catalog_sync.py` 失败，`grok-imagine-1.0-fast` 仅存在于 `model.py` 而不在 `models.ts`
- **方案**: 在 `src/grok/models.ts` 中添加对应的 `grok-imagine-1.0-fast` 条目
- **修改文件**: `src/grok/models.ts`

### B8: 配置迁移系统增强
- **问题**: tqzhr 配置系统缺少废弃键迁移和未知键修剪
- **方案**: 添加 _migrate_deprecated_config 和 _prune_unknown_config 函数，集成到 load/update 流程
- **修改文件**: `app/core/config.py`
