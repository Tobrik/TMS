"""Tests for llm.py: model fallback, schema fallback, reasoning params."""

import asyncio

import httpx

import llm

MESSAGES = [{"role": "user", "content": "hi"}]
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
          "additionalProperties": False}


def test_no_api_key_returns_none_without_calling(fake_groq, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    assert asyncio.run(llm.chat_json(MESSAGES, SCHEMA, name="t")) is None
    assert fake_groq.calls == []


def test_falls_back_to_next_model(fake_groq):
    first, second = llm.TEXT_MODELS[:2]
    fake_groq.replies["t"] = lambda p: 503 if p["model"] == first else {"ok": True}
    result = asyncio.run(llm.chat_json(MESSAGES, SCHEMA, name="t"))
    assert result.data == {"ok": True}
    assert result.model == second


def test_rejected_schema_retries_same_model_in_json_mode(fake_groq):
    fake_groq.replies["t"] = 400
    fake_groq.replies["json_object"] = {"ok": True}
    result = asyncio.run(llm.chat_json(MESSAGES, SCHEMA, name="t"))
    assert result.data == {"ok": True}
    assert fake_groq.purposes() == ["t", "json_object"]
    assert fake_groq.calls[0][1] == fake_groq.calls[1][1]


def test_all_models_fail_returns_none(fake_groq):
    fake_groq.replies["t"] = 500
    assert asyncio.run(llm.chat_json(MESSAGES, SCHEMA, name="t")) is None
    assert len(fake_groq.calls) == len(llm.TEXT_MODELS)


def test_rate_limit_waits_and_retries_same_model(fake_groq):
    first = llm.TEXT_MODELS[0]
    replies = iter([httpx.Response(429, headers={"retry-after": "0"}, text="slow down"), {"ok": True}])
    fake_groq.replies["t"] = lambda _payload: next(replies)
    result = asyncio.run(llm.chat_json(MESSAGES, SCHEMA, name="t"))
    assert result.data == {"ok": True}
    assert [model for _, model, _ in fake_groq.calls] == [first, first]


def test_retries_up_to_max_retries(fake_groq, monkeypatch):
    monkeypatch.setattr(llm, "MAX_RETRIES", 3)
    first = llm.TEXT_MODELS[0]
    replies = iter([httpx.Response(429, headers={"retry-after": "0"}, text="slow down")] * 3 + [{"ok": True}])
    fake_groq.replies["t"] = lambda p: next(replies) if p["model"] == first else 500
    assert asyncio.run(llm.chat_json(MESSAGES, SCHEMA, name="t")).model == first


def test_long_rate_limit_moves_to_next_model(fake_groq, monkeypatch):
    first, second = llm.TEXT_MODELS[:2]
    monkeypatch.setattr(llm, "MAX_RETRY_WAIT", 1.0)
    fake_groq.replies["t"] = lambda p: (
        httpx.Response(429, headers={"retry-after": "60"}, text="slow down") if p["model"] == first else {"ok": True})
    assert asyncio.run(llm.chat_json(MESSAGES, SCHEMA, name="t")).model == second


def test_invalid_json_moves_on(fake_groq):
    first = llm.TEXT_MODELS[0]
    fake_groq.replies["t"] = lambda p: "not json" if p["model"] == first else {"ok": True}
    assert asyncio.run(llm.chat_json(MESSAGES, SCHEMA, name="t")).data == {"ok": True}


def test_code_fences_are_stripped(fake_groq):
    fake_groq.replies["json_object"] = '```json\n{"ok": true}\n```'
    result = asyncio.run(llm.chat_json(MESSAGES, SCHEMA, name="t", strict=False))
    assert result.data == {"ok": True}


def test_reasoning_params_per_model_family(fake_groq):
    fake_groq.replies["text"] = "hello"
    asyncio.run(llm.chat_text(MESSAGES, name="t", models=["openai/gpt-oss-120b"], reasoning="medium"))
    asyncio.run(llm.chat_text(MESSAGES, name="t", models=["qwen/qwen3.8-27b"], reasoning="none"))
    gpt, qwen = fake_groq.calls[0][2], fake_groq.calls[1][2]
    assert gpt["reasoning_effort"] == "medium" and gpt["include_reasoning"] is False
    assert "reasoning_format" not in gpt
    assert qwen["reasoning_effort"] == "none" and qwen["reasoning_format"] == "hidden"
    assert "include_reasoning" not in qwen
