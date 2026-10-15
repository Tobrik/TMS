"""
Single entry point for every LLM call (Groq, OpenAI-compatible API).

- Models come from env, as ordered fallback lists. Groq retired llama-3.3-70b
  and llama-4-scout in 2026; swapping models must be a config change.
    LLM_TEXT_MODELS    default "openai/gpt-oss-120b,openai/gpt-oss-20b"
    LLM_VISION_MODELS  default "qwen/qwen3.8-27b"  (Groq Preview tier — the only
                       vision model it serves; may be retired at short notice)
- JSON calls use strict json_schema (constrained decoding). If a model rejects
  the schema (HTTP 400) the same model is retried in json_object mode; the caller
  validates the result either way.
- Reasoning: gpt-oss always reasons (low/medium/high) — we hide it from the
  response; qwen supports "none". Reasoning text is never shown to anyone.
- Returns None when every model fails: callers must have a degraded path.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import httpx

logger = logging.getLogger("tms.llm")

BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
TEXT_MODELS = [m.strip() for m in os.environ.get(
    "LLM_TEXT_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b").split(",") if m.strip()]
VISION_MODELS = [m.strip() for m in os.environ.get(
    "LLM_VISION_MODELS", "qwen/qwen3.8-27b").split(",") if m.strip()]

# On HTTP 429, wait for Groq's retry-after (if no longer than MAX_RETRY_WAIT) and
# retry the same model up to MAX_RETRIES times before moving on. Batch evals
# raise both to ride out the per-minute token limit.
MAX_RETRY_WAIT = float(os.environ.get("LLM_MAX_RETRY_WAIT", "5"))
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "1"))

# Tests replace this with an httpx.MockTransport.
_transport: Optional[httpx.AsyncBaseTransport] = None


def _retry_after(resp: httpx.Response) -> Optional[float]:
    try:
        return float(resp.headers.get("retry-after", ""))
    except ValueError:
        return None


@dataclass
class LLMResult:
    data: Any        # parsed JSON (chat_json) or str (chat_text)
    model: str
    input_tokens: int = 0
    output_tokens: int = 0   # includes hidden reasoning tokens (billed as output)


def _api_key() -> str:
    return os.environ.get("GROQ_API_KEY", "")


def available() -> bool:
    return bool(_api_key())


def _reasoning_params(model: str, effort: str) -> dict:
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low" if effort == "none" else effort, "include_reasoning": False}
    if model.startswith("qwen/"):
        return {"reasoning_effort": effort, "reasoning_format": "hidden"}
    return {}


def _strip_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3]
    return content.strip()


async def _post(client: httpx.AsyncClient, payload: dict) -> httpx.Response:
    return await client.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
        json=payload,
    )


async def _complete(
    messages: List[dict],
    *,
    models: List[str],
    response_format: Optional[dict],
    reasoning: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    purpose: str,
    parse_json: bool,
) -> Optional[LLMResult]:
    if not available():
        logger.warning("LLM %s skipped: GROQ_API_KEY not set", purpose)
        return None

    async with httpx.AsyncClient(timeout=timeout, transport=_transport) as client:
        for model in models:
            formats = [response_format]
            if response_format and response_format.get("type") == "json_schema":
                formats.append({"type": "json_object"})  # fallback if the schema is rejected

            for fmt in formats:
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                    **_reasoning_params(model, reasoning),
                }
                if fmt:
                    payload["response_format"] = fmt
                started = time.perf_counter()
                try:
                    resp = await _post(client, payload)
                    for _ in range(MAX_RETRIES):
                        wait = _retry_after(resp) if resp.status_code == 429 else None
                        if wait is None or wait > MAX_RETRY_WAIT:
                            break
                        logger.info("LLM %s %s: rate-limited, retrying in %.1f s", purpose, model, wait)
                        await asyncio.sleep(wait)
                        resp = await _post(client, payload)
                except httpx.HTTPError as e:
                    logger.warning("LLM %s %s: request failed (%s)", purpose, model, type(e).__name__)
                    break  # next model
                elapsed = (time.perf_counter() - started) * 1000

                if resp.status_code == 400 and fmt and fmt.get("type") == "json_schema":
                    logger.warning("LLM %s %s: schema rejected (%s), retrying in json_object mode",
                                   purpose, model, resp.text[:200])
                    continue  # same model, next format
                if resp.status_code != 200:
                    logger.warning("LLM %s %s: HTTP %d %s", purpose, model, resp.status_code, resp.text[:200])
                    break  # next model

                body = resp.json()
                content = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                usage = body.get("usage", {})
                logger.info("LLM %s %s: %.0f ms, %s in / %s out tokens", purpose, model, elapsed,
                            usage.get("prompt_tokens"), usage.get("completion_tokens"))
                tokens = (usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0)
                if not parse_json:
                    if content.strip():
                        return LLMResult(content.strip(), model, *tokens)
                    break
                try:
                    return LLMResult(json.loads(_strip_fences(content)), model, *tokens)
                except json.JSONDecodeError:
                    logger.warning("LLM %s %s: invalid JSON in response", purpose, model)
                    break
    logger.error("LLM %s: all models failed (%s)", purpose, ", ".join(models))
    return None


async def chat_json(
    messages: List[dict],
    schema: dict,
    *,
    name: str,
    models: Optional[List[str]] = None,
    strict: bool = True,
    reasoning: str = "low",
    temperature: float = 0.6,
    max_tokens: int = 4000,
    timeout: float = 30.0,
) -> Optional[LLMResult]:
    """JSON completion. With strict=False (e.g. image input, where Groq doesn't
    document structured outputs) uses json_object mode; the schema then only
    guides the prompt, so validate the result."""
    response_format = (
        {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
        if strict else {"type": "json_object"}
    )
    return await _complete(
        messages, models=models or TEXT_MODELS, response_format=response_format, reasoning=reasoning,
        temperature=temperature, max_tokens=max_tokens, timeout=timeout, purpose=name, parse_json=True,
    )


async def chat_text(
    messages: List[dict],
    *,
    name: str,
    models: Optional[List[str]] = None,
    reasoning: str = "low",
    temperature: float = 0.5,
    max_tokens: int = 2000,
    timeout: float = 30.0,
) -> Optional[LLMResult]:
    return await _complete(
        messages, models=models or TEXT_MODELS, response_format=None, reasoning=reasoning,
        temperature=temperature, max_tokens=max_tokens, timeout=timeout, purpose=name, parse_json=False,
    )
