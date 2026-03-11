"""
Chat Completions API 路由
"""

from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.core.auth import verify_api_key
from app.services.grok.chat import ChatService
from app.services.grok.model import ModelService
from app.services.grok.tool_call import build_tool_prompt, parse_tool_calls, format_tool_history
from app.core.exceptions import ValidationException
from app.services.quota import enforce_daily_quota


router = APIRouter(tags=["Chat"])


VALID_ROLES = ["developer", "system", "user", "assistant", "tool"]
USER_CONTENT_TYPES = ["text", "image_url", "input_audio", "file"]


class MessageItem(BaseModel):
    """消息项"""
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    
    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}")
        return v


class VideoConfig(BaseModel):
    """视频生成配置"""
    aspect_ratio: Optional[str] = Field("3:2", description="视频比例: 3:2, 16:9, 1:1 等")
    video_length: Optional[int] = Field(6, description="视频时长(秒): 5-15")
    resolution: Optional[str] = Field("SD", description="视频分辨率: SD, HD")
    preset: Optional[str] = Field("custom", description="风格预设: fun, normal, spicy")
    
    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v):
        allowed = ["2:3", "3:2", "1:1", "9:16", "16:9"]
        if v and v not in allowed:
            raise ValidationException(
                message=f"aspect_ratio must be one of {allowed}",
                param="video_config.aspect_ratio",
                code="invalid_aspect_ratio"
            )
        return v
    
    @field_validator("video_length")
    @classmethod
    def validate_video_length(cls, v):
        if v is not None:
            if v < 5 or v > 15:
                raise ValidationException(
                    message="video_length must be between 5 and 15 seconds",
                    param="video_config.video_length",
                    code="invalid_video_length"
                )
        return v

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v):
        allowed = ["SD", "HD"]
        if v and v not in allowed:
            raise ValidationException(
                message=f"resolution must be one of {allowed}",
                param="video_config.resolution",
                code="invalid_resolution"
            )
        return v
    
    @field_validator("preset")
    @classmethod
    def validate_preset(cls, v):
        # 允许为空，默认 custom
        if not v:
            return "custom"
        allowed = ["fun", "normal", "spicy", "custom"]
        if v not in allowed:
             raise ValidationException(
                message=f"preset must be one of {allowed}",
                param="video_config.preset",
                code="invalid_preset"
             )
        return v


class ImageConfig(BaseModel):
    """图片生成配置"""
    n: Optional[int] = Field(1, ge=1, le=10, description="生成数量 (1-10)")
    size: Optional[str] = Field("1024x1024", description="图片尺寸")
    response_format: Optional[str] = Field(None, description="响应格式")


class ChatCompletionRequest(BaseModel):
    """Chat Completions 请求"""
    model: str = Field(..., description="模型名称")
    messages: List[MessageItem] = Field(..., description="消息数组")
    stream: Optional[bool] = Field(None, description="是否流式输出")
    thinking: Optional[str] = Field(None, description="思考模式: enabled/disabled/None")
    reasoning_effort: Optional[str] = Field(None, description="推理强度: none/minimal/low/medium/high/xhigh")
    temperature: Optional[float] = Field(0.8, description="采样温度: 0-2")
    top_p: Optional[float] = Field(0.95, description="nucleus 采样: 0-1")
    
    # 视频生成配置
    video_config: Optional[VideoConfig] = Field(None, description="视频生成参数")
    # 图片生成配置
    image_config: Optional[ImageConfig] = Field(None, description="图片生成参数")
    # Tool calling
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Tool definitions")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Tool choice: auto/required/none/specific")
    parallel_tool_calls: Optional[bool] = Field(True, description="Allow parallel tool calls")
    
    model_config = {
        "extra": "ignore"
    }


def validate_request(request: ChatCompletionRequest):
    """验证请求参数"""
    # 验证模型
    if not ModelService.valid(request.model):
        raise ValidationException(
            message=f"The model `{request.model}` does not exist or you do not have access to it.",
            param="model",
            code="model_not_found"
        )
    
    # 验证消息
    for idx, msg in enumerate(request.messages):
        # tool role: requires tool_call_id, content can be None/empty
        if msg.role == "tool":
            if not msg.tool_call_id:
                raise ValidationException(
                    message="tool messages must have a 'tool_call_id' field",
                    param=f"messages.{idx}.tool_call_id",
                    code="missing_tool_call_id"
                )
            continue

        # assistant with tool_calls: content can be None
        if msg.role == "assistant" and msg.tool_calls:
            continue

        content = msg.content
        
        # 字符串内容
        if isinstance(content, str):
            if not content.strip():
                raise ValidationException(
                    message="Message content cannot be empty",
                    param=f"messages.{idx}.content",
                    code="empty_content"
                )
        
        # 列表内容
        elif isinstance(content, list):
            if not content:
                raise ValidationException(
                    message="Message content cannot be an empty array",
                    param=f"messages.{idx}.content",
                    code="empty_content"
                )
            
            for block_idx, block in enumerate(content):
                # 检查空对象
                if not block:
                    raise ValidationException(
                        message="Content block cannot be empty",
                        param=f"messages.{idx}.content.{block_idx}",
                        code="empty_block"
                    )
                
                # 检查 type 字段
                if "type" not in block:
                    raise ValidationException(
                        message="Content block must have a 'type' field",
                        param=f"messages.{idx}.content.{block_idx}",
                        code="missing_type"
                    )
                
                block_type = block.get("type")
                
                # 检查 type 空值
                if not block_type or not isinstance(block_type, str) or not block_type.strip():
                    raise ValidationException(
                        message="Content block 'type' cannot be empty",
                        param=f"messages.{idx}.content.{block_idx}.type",
                        code="empty_type"
                    )
                
                # 验证 type 有效性
                if msg.role == "user":
                    if block_type not in USER_CONTENT_TYPES:
                        raise ValidationException(
                            message=f"Invalid content block type: '{block_type}'",
                            param=f"messages.{idx}.content.{block_idx}.type",
                            code="invalid_type"
                        )
                elif block_type != "text":
                    raise ValidationException(
                        message=f"The `{msg.role}` role only supports 'text' type, got '{block_type}'",
                        param=f"messages.{idx}.content.{block_idx}.type",
                        code="invalid_type"
                    )
                
                # 验证字段是否存在 & 非空
                if block_type == "text":
                    text = block.get("text", "")
                    if not isinstance(text, str) or not text.strip():
                        raise ValidationException(
                            message="Text content cannot be empty",
                            param=f"messages.{idx}.content.{block_idx}.text",
                            code="empty_text"
                        )
                elif block_type == "image_url":
                    image_url = block.get("image_url")
                    if not image_url or not (isinstance(image_url, dict) and image_url.get("url")):
                        raise ValidationException(
                            message="image_url must have a 'url' field",
                            param=f"messages.{idx}.content.{block_idx}.image_url",
                            code="missing_url"
                        )


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key: Optional[str] = Depends(verify_api_key)):
    """Chat Completions API - 兼容 OpenAI"""
    
    # 参数验证
    validate_request(request)

    # Daily quota (best-effort)
    await enforce_daily_quota(api_key, request.model)
    
    # Tool calling 标记（视频模型不支持 tool calling）
    has_tools = bool(request.tools)

    # 检测模型类型
    model_info = ModelService.get(request.model)

    if model_info and model_info.is_video:
        has_tools = False
        from app.services.grok.media import VideoService
        
        # 提取视频配置 (默认值在 Pydantic 模型中处理)
        v_conf = request.video_config or VideoConfig()
        
        result = await VideoService.completions(
            model=request.model,
            messages=[msg.model_dump() for msg in request.messages],
            stream=request.stream,
            thinking=request.thinking,
            aspect_ratio=v_conf.aspect_ratio,
            video_length=v_conf.video_length,
            resolution=v_conf.resolution,
            preset=v_conf.preset
        )

    elif model_info and model_info.is_image:
        has_tools = False
        from app.api.v1.image import (
            call_grok_legacy, resolve_aspect_ratio, resolve_response_format,
            response_field_name, _get_token_for_model, _pick_images, _dedupe_images,
            _image_generation_method, _is_valid_image_value,
            _collect_experimental_generation_images,
            IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL,
        )
        from app.services.grok.processor import ImageStreamProcessor
        import time as _time

        # 从消息中提取 prompt
        prompt = ""
        for msg in reversed(request.messages):
            c = msg.content
            if isinstance(c, str) and c.strip():
                prompt = c.strip()
                break
            if isinstance(c, list):
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block.get("text", "")
                        if isinstance(t, str) and t.strip():
                            prompt = t.strip()
                if prompt:
                    break

        if not prompt:
            raise ValidationException(
                message="Prompt cannot be empty for image generation",
                param="messages",
                code="empty_prompt",
            )

        # 图片配置
        img_conf = request.image_config or ImageConfig()
        n = img_conf.n or 1
        size = img_conf.size or "1024x1024"
        aspect_ratio = resolve_aspect_ratio(size)
        response_format = resolve_response_format(img_conf.response_format)
        resp_field = response_field_name(response_format)

        token_mgr, token = await _get_token_for_model(request.model)
        image_method = _image_generation_method()

        is_stream = request.stream if request.stream is not None else False

        if is_stream:
            # 流式图片生成 → 包装为 chat completion SSE
            from app.services.grok.chat import GrokChatService
            chat_service = GrokChatService()
            response = await chat_service.chat(
                token=token,
                message=f"Image Generation: {prompt}",
                model=model_info.grok_model,
                mode=model_info.model_mode,
                think=False,
                stream=True,
            )
            processor = ImageStreamProcessor(
                model_info.model_id,
                token,
                n=n,
                response_format=response_format,
            )

            async def _img_chat_stream():
                completed = False
                try:
                    async for chunk in processor.process(response):
                        yield chunk
                    completed = True
                finally:
                    try:
                        if completed:
                            await token_mgr.sync_usage(token, model_info.model_id, consume_on_fail=True, is_usage=True)
                    except Exception:
                        pass

            result = _img_chat_stream()
        else:
            # 非流式：收集图片，包装为 chat completion 格式
            all_images = []
            if image_method == IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL:
                try:
                    all_images = await _collect_experimental_generation_images(
                        token=token, prompt=prompt, n=n,
                        response_format=response_format,
                        aspect_ratio=aspect_ratio, concurrency=1,
                    )
                except Exception:
                    pass

            if not all_images:
                all_images = await call_grok_legacy(
                    token, f"Image Generation: {prompt}",
                    model_info, response_format=response_format,
                )

            selected = _pick_images(_dedupe_images(all_images), n)
            # 构造 markdown 内容
            content_parts = []
            for i, img in enumerate(selected):
                if _is_valid_image_value(img):
                    if response_format == "url":
                        content_parts.append(f"![image-{i}]({img})")
                    else:
                        content_parts.append(f"![image-{i}]({img})")
                else:
                    content_parts.append("[Image generation failed]")

            result = {
                "id": f"chatcmpl-img-{_time.time_ns()}",
                "object": "chat.completion",
                "created": int(_time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "\n".join(content_parts)},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    else:
        # reasoning_effort 优先于 thinking
        thinking = request.thinking
        if request.reasoning_effort:
            effort = request.reasoning_effort.lower()
            if effort in ("none", "minimal"):
                thinking = "disabled"
            elif effort in ("low", "medium", "high", "xhigh"):
                thinking = "enabled"

        # Tool calling: 预处理消息
        messages = [msg.model_dump() for msg in request.messages]
        has_tools = bool(request.tools)
        if has_tools:
            # 将 tool 角色消息转换为文本格式
            messages = format_tool_history(messages)
            # 注入 tool prompt 作为 system 消息
            tool_prompt = build_tool_prompt(
                request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls or True,
            )
            if tool_prompt:
                messages.insert(0, {"role": "system", "content": tool_prompt})

        result = await ChatService.completions(
            model=request.model,
            messages=messages,
            stream=request.stream,
            thinking=thinking
        )
    
    if isinstance(result, dict):
        # 非流式模式：后处理解析 tool_calls
        if has_tools and isinstance(result, dict):
            choices = result.get("choices", [])
            if choices and isinstance(choices[0], dict):
                msg_obj = choices[0].get("message", {})
                content = msg_obj.get("content", "")
                if content:
                    text_content, tool_calls_list = parse_tool_calls(content, request.tools)
                    if tool_calls_list:
                        msg_obj["tool_calls"] = tool_calls_list
                        msg_obj["content"] = text_content
                        choices[0]["finish_reason"] = "tool_calls"
        return JSONResponse(content=result)
    else:
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )


__all__ = ["router"]
