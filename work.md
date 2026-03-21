# Grok2API-tqzhr 功能同步记录

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

## 2026-03-22 — 修复生图功能（ws_imagine → app-chat REST API 迁移）

### 问题
- Grok 官方废弃了 WebSocket 端点 `wss://grok.com/ws/imagine/listen`，生图功能迁移到 app-chat REST API
- grok2api 生图代码仍调用废弃的 WS 端点，导致所有生图请求返回 `rate_limit_exceeded`

### 方案
- 改用 app-chat REST API + `imageGen` tool override 触发生图，保留 ws_imagine 作为 fallback

### 修改文件
1. **`app/services/grok/chat.py`**
   - `build_payload()` 新增 `request_overrides` 参数，允许覆盖 payload 字段
   - `GrokChatService.chat()` 新增 `request_overrides` 参数并透传
2. **`app/services/grok/imagine_experimental.py`**
   - 新增 `_build_app_chat_payload()` 静态方法，构造生图 request_overrides
   - 新增 `generate_app_chat()` 异步方法，通过 app-chat REST API 触发生图
   - 新增 `IMAGE_METHOD_APP_CHAT` 常量和别名映射
3. **`app/services/grok/imagine_generation.py`**
   - 新增 `call_app_chat_generation_once()` 方法（app-chat + ImageCollectProcessor）
   - 新增 `call_ws_generation_once()` 方法（原 ws 路径，作为 fallback）
   - 修改 `call_experimental_generation_once()` 优先走 app-chat，失败 fallback 到 ws
4. **`app/api/v1/image.py`**
   - 修改 `_experimental_stream_generation()` 优先走 app-chat + ImageStreamProcessor 流式生图，失败 fallback 到 ws

5. **`src/grok/imagineExperimental.ts`**
   - 新增 `IMAGE_METHOD_APP_CHAT` 常量和别名映射
   - 新增 `buildAppChatImageGenPayload()` 函数，构造 app-chat 生图 payload
   - 新增 `generateImagineAppChat()` 导出函数，通过 app-chat REST API 触发生图
6. **`src/routes/openai.ts`**
   - 导入 `generateImagineAppChat`
   - 修改 `collectExperimentalGenerationImages()` 优先走 app-chat REST API，失败 fallback 到 ws

### 撤回方式
- `git checkout HEAD -- app/services/grok/chat.py app/services/grok/imagine_experimental.py app/services/grok/imagine_generation.py app/api/v1/image.py src/grok/imagineExperimental.ts src/routes/openai.ts`
