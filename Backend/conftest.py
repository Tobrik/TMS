"""Shared pytest fixtures. `fake_groq` replaces the Groq API for every LLM call."""

import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import clef  # noqa: E402
import llm  # noqa: E402


class FakeGroq:
    """
    Answers chat completions by purpose: the json_schema name ("extract",
    "review", ...), "json_object", or "text". A reply is a dict/str (returned as
    the message content), an int (HTTP error status), an httpx.Response (returned
    as is), or a callable(payload) returning any of these.
    """

    def __init__(self):
        self.replies = {}
        self.calls = []  # (purpose, model, payload)

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        fmt = payload.get("response_format") or {}
        purpose = fmt["json_schema"]["name"] if fmt.get("type") == "json_schema" else fmt.get("type", "text")
        self.calls.append((purpose, payload["model"], payload))

        reply = self.replies.get(purpose)
        if callable(reply):
            reply = reply(payload)
        if reply is None:
            return httpx.Response(500, text=f"no fake reply for {purpose}")
        if isinstance(reply, httpx.Response):
            return reply
        if isinstance(reply, int):
            return httpx.Response(reply, text=f"fake error {reply}")
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    def purposes(self):
        return [purpose for purpose, _, _ in self.calls]


@pytest.fixture
def fake_groq(monkeypatch):
    fake = FakeGroq()
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_transport", httpx.MockTransport(fake.handler))
    return fake


# ─── Fake Cloudflare Workers AI (Clef) ───────────────────────────────────────

def status(reported=0.0, denied=0.0):
    return {"type": "choice", "choice": "x", "confidence": 0.9,
            "probabilities": {"not_mentioned": round(1 - reported - denied, 3), "reported": reported, "denied": denied}}


def choice(probabilities):
    return {"type": "choice", "choice": max(probabilities, key=probabilities.get), "confidence": 0.9,
            "probabilities": probabilities}


@pytest.fixture
def fake_clef(monkeypatch):
    """Answers every question 'not mentioned' unless overridden in `answers`."""
    monkeypatch.setenv("CF_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CF_API_TOKEN", "token")
    state = {"answers": {}, "requests": [], "fail": False}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        state["requests"].append((request.url.path, body))
        if state["fail"] and len(state["requests"]) == 1:
            return httpx.Response(500, text="boom")
        answers = {}
        for qid, q in body["questions"].items():
            default = {"type": "noul", "noul": 0.02} if q["type"] == "noul" else choice(
                {opt: (1.0 if i == 0 else 0.0) for i, opt in enumerate(q["criteria"])})
            answers[qid] = state["answers"].get(qid, default)
        return httpx.Response(200, json={"result": {"model": body["model"], "answers": answers,
                                                    "usage": {"input_tokens": 2000, "output_tokens": 0}},
                                         "success": True, "errors": []})

    monkeypatch.setattr(clef, "_transport", httpx.MockTransport(handler))
    return state
