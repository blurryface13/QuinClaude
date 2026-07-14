from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import httpx

from quin_claude.core.bus.events import LlmModelSelectedEvent, LlmTokenEvent, LlmUsageEvent
from quin_claude.core.events.bus import EventBus
from quin_claude.core.llm.types import LlmResponse, ToolCallBlock, UsageStats

_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-opus-4-7": 200_000,
    "deepseek-chat": 64_000,
    "deepseek-reasoner": 64_000,
}

_MAX_STREAM_RETRIES = 3
_RETRY_BACKOFF_S = (1.0, 2.0, 4.0)

log = logging.getLogger(__name__)


# 返回指定模型的最大 context window token 数
def _context_window(model: str) -> int:
    return _MODEL_CONTEXT_WINDOWS.get(model, 200_000)


_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Use the available tools to complete the user's goal. "
    "When the goal is fully achieved, respond with a final answer and do not call any more tools."
)


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


class AnthropicProvider:
    # 初始化 Anthropic 客户端；client 可在测试时注入以跳过 API key 检查
    def __init__(self, model: str, client: Any = None) -> None:
        self._client: Any = client
        self._model = model

    # 流式调用 Anthropic API，逐 token 发布事件并返回 LlmResponse；网络中断时自动重试
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=api_key)

        await bus.publish(
            LlmModelSelectedEvent(run_id=run_id, model=self._model, strategy="static", ts=_now())
        )

        system_blocks: list[dict[str, object]] = [
            {
                "type": "text",
                "text": system or _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
        ]

        tools: list[dict[str, object]] = list(tool_schemas)
        if tools:
            last = dict(tools[-1])
            last["cache_control"] = {"type": "ephemeral"}
            tools = tools[:-1] + [last]

        kwargs: dict[str, object] = {
            "model": self._model,
            "max_tokens": 8192,
            "system": system_blocks,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        text_parts: list[str] = []
        final_message: Any = None

        for attempt in range(1, _MAX_STREAM_RETRIES + 1):
            text_parts = []
            try:
                async with self._client.messages.stream(**kwargs) as stream:
                    async for text in stream.text_stream:
                        # Only publish token events on the first attempt to avoid TUI duplicates
                        if attempt == 1:
                            await bus.publish(LlmTokenEvent(run_id=run_id, token=text, ts=_now()))
                        text_parts.append(text)
                    final_message = await stream.get_final_message()
                break  # success
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as exc:
                if attempt == _MAX_STREAM_RETRIES:
                    log.error(
                        "stream failed after %d attempts run_id=%s step=%d: %s",
                        _MAX_STREAM_RETRIES, run_id, step, exc,
                    )
                    raise
                delay = _RETRY_BACKOFF_S[attempt - 1]
                log.warning(
                    "stream dropped (attempt %d/%d) run_id=%s step=%d: %s — retrying in %.0fs",
                    attempt, _MAX_STREAM_RETRIES, run_id, step, exc, delay,
                )
                await asyncio.sleep(delay)

        assert final_message is not None

        usage = final_message.usage
        cache_read: int = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create: int = getattr(usage, "cache_creation_input_tokens", 0) or 0
        context_pct = usage.input_tokens / _context_window(self._model)

        await bus.publish(
            LlmUsageEvent(
                run_id=run_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_create,
                context_pct=context_pct,
                ts=_now(),
            )
        )

        tool_calls: list[ToolCallBlock] = []
        thinking_blocks: list[dict[str, object]] = []
        for block in final_message.content:
            if block.type == "tool_use":
                tool_calls.append(
                    ToolCallBlock(id=block.id, name=block.name, input=dict(block.input))
                )
            elif block.type == "thinking":
                # thinking blocks must be passed back verbatim in subsequent requests
                thinking_blocks.append({"type": "thinking", "thinking": block.thinking, "signature": block.signature})

        return LlmResponse(
            stop_reason=final_message.stop_reason or "end_turn",
            tool_calls=tool_calls,
            text="".join(text_parts),
            thinking_blocks=thinking_blocks,
            usage=UsageStats(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_create,
                context_pct=context_pct,
            ),
        )


class OpenAICompatibleProvider:
    """Streaming provider for DeepSeek and other OpenAI-compatible chat APIs."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str,
        provider_name: str = "openai_compatible",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._provider_name = provider_name
        self._client = client

    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        if not self._api_key:
            raise RuntimeError("QUIN_LLM_API_KEY not set")

        await bus.publish(
            LlmModelSelectedEvent(run_id=run_id, model=self._model, strategy="static", ts=_now())
        )

        payload: dict[str, object] = {
            "model": self._model,
            "messages": _to_openai_messages(messages, system or _SYSTEM_PROMPT),
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 8192,
        }
        tools = _to_openai_tools(tool_schemas)
        if tools:
            payload["tools"] = tools

        text_parts: list[str] = []
        tool_parts: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage_dict: dict[str, int] = {}

        async def _run_stream(client: httpx.AsyncClient) -> None:
            nonlocal finish_reason, usage_dict
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=120.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line.removeprefix("data: ").strip()
                    if not raw or raw == "[DONE]":
                        continue
                    chunk = json.loads(raw)
                    if chunk.get("usage"):
                        usage_dict = dict(chunk["usage"])
                    for choice in chunk.get("choices", []):
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            text = str(content)
                            text_parts.append(text)
                            await bus.publish(LlmTokenEvent(run_id=run_id, token=text, ts=_now()))
                        for tool_delta in delta.get("tool_calls") or []:
                            idx = int(tool_delta.get("index", 0))
                            part = tool_parts.setdefault(
                                idx, {"id": "", "name": "", "arguments": []}
                            )
                            if tool_delta.get("id"):
                                part["id"] = str(tool_delta["id"])
                            fn = tool_delta.get("function") or {}
                            if fn.get("name"):
                                part["name"] = str(fn["name"])
                            if fn.get("arguments"):
                                part["arguments"].append(str(fn["arguments"]))

        for attempt in range(1, _MAX_STREAM_RETRIES + 1):
            try:
                if self._client is not None:
                    await _run_stream(self._client)
                else:
                    async with httpx.AsyncClient() as client:
                        await _run_stream(client)
                break
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as exc:
                if attempt == _MAX_STREAM_RETRIES:
                    log.error(
                        "%s stream failed after %d attempts run_id=%s step=%d: %s",
                        self._provider_name,
                        _MAX_STREAM_RETRIES,
                        run_id,
                        step,
                        exc,
                    )
                    raise
                await asyncio.sleep(_RETRY_BACKOFF_S[attempt - 1])

        tool_calls = [
            ToolCallBlock(
                id=part["id"] or f"tool_call_{idx}",
                name=part["name"],
                input=_loads_tool_arguments("".join(part["arguments"])),
            )
            for idx, part in sorted(tool_parts.items())
            if part["name"]
        ]

        input_tokens = int(usage_dict.get("prompt_tokens", 0))
        output_tokens = int(usage_dict.get("completion_tokens", 0))
        context_pct = input_tokens / _context_window(self._model) if input_tokens else 0.0
        await bus.publish(
            LlmUsageEvent(
                run_id=run_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                context_pct=context_pct,
                ts=_now(),
            )
        )

        return LlmResponse(
            stop_reason=_map_finish_reason(finish_reason, tool_calls),
            tool_calls=tool_calls,
            text="".join(text_parts),
            usage=UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                context_pct=context_pct,
            ),
        )


def create_provider(config: Any) -> Any:
    provider = config.llm.provider.lower()
    if provider == "anthropic":
        return AnthropicProvider(config.llm.default_model)
    if provider in {"deepseek", "openai_compatible", "openai-compatible"}:
        return OpenAICompatibleProvider(
            config.llm.default_model,
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            provider_name=provider,
        )
    raise SystemExit(f"Config error: unsupported llm.provider {config.llm.provider!r}")


def _to_openai_tools(tool_schemas: list[dict[str, object]]) -> list[dict[str, object]]:
    tools: list[dict[str, object]] = []
    for schema in tool_schemas:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": schema.get("name", ""),
                    "description": schema.get("description", ""),
                    "parameters": schema.get("input_schema", {"type": "object"}),
                },
            }
        )
    return tools


def _to_openai_messages(
    messages: list[dict[str, object]], system_prompt: str
) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict[str, object]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": str(block.get("id", "")),
                            "type": "function",
                            "function": {
                                "name": str(block.get("name", "")),
                                "arguments": json.dumps(
                                    block.get("input", {}), ensure_ascii=False
                                ),
                            },
                        }
                    )
            assistant: dict[str, object] = {
                "role": "assistant",
                "content": "".join(text_parts) or None,
            }
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            converted.append(assistant)
        elif role == "user" and isinstance(content, list):
            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(block.get("tool_use_id", "")),
                            "content": str(block.get("content", "")),
                        }
                    )
                elif block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
            if text_parts:
                converted.append({"role": "user", "content": "\n".join(text_parts)})
        else:
            converted.append({"role": role, "content": _content_to_text(content)})
    return converted


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _loads_tool_arguments(raw: str) -> dict[str, object]:
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return data if isinstance(data, dict) else {"value": data}


def _map_finish_reason(reason: str, tool_calls: list[ToolCallBlock]) -> str:
    if tool_calls or reason == "tool_calls":
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    return "end_turn"
