"""
Cloudflare Clef / Clef-flash decision models via the Workers AI REST API.

EXPERIMENTAL — used by extraction_clef.py and eval/compare_extraction.py to
compare against the Groq pipeline; not wired into the API.

A request is a `state` (the patient's text) plus up to 64 typed questions:
  noul   → probability of yes
  choice → 2-255 named options, probability per option
  score  → 2-10 ordered levels, probability per level
No text is generated. Billing is input tokens only (2026-10: clef-flash
$0.038/M, clef $0.24/M); every account gets 10,000 free neurons/day
(~2.9M clef-flash tokens, ~0.46M clef tokens).

Env: CF_ACCOUNT_ID, CF_API_TOKEN (a token with Workers AI permission),
     CLEF_MODEL = clef-flash (default) | clef
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger("tms.clef")

URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"
MAX_QUESTIONS = 64
MAX_RETRY_WAIT = float(os.environ.get("CLEF_MAX_RETRY_WAIT", "5"))

# Tests replace this with an httpx.MockTransport.
_transport: Optional[httpx.AsyncBaseTransport] = None


@dataclass
class ClefResult:
    answers: dict      # question id -> {"type", "noul" | "choice"+"probabilities" | "score"+...}
    model: str
    input_tokens: int = 0


def available() -> bool:
    return bool(os.environ.get("CF_ACCOUNT_ID") and os.environ.get("CF_API_TOKEN"))


def default_model() -> str:
    return os.environ.get("CLEF_MODEL", "clef-flash")


async def ask(state, questions: dict, *, model: Optional[str] = None, timeout: float = 60.0) -> Optional[ClefResult]:
    """One Clef request. Returns None on any failure (callers degrade)."""
    model = model or default_model()
    if not available():
        logger.warning("Clef skipped: CF_ACCOUNT_ID / CF_API_TOKEN not set")
        return None
    if not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f"Clef takes 1-{MAX_QUESTIONS} questions, got {len(questions)}")

    url = URL.format(account=os.environ["CF_ACCOUNT_ID"], model=model)
    headers = {"Authorization": f"Bearer {os.environ['CF_API_TOKEN']}"}
    body = {"model": model, "state": state, "questions": questions}

    async with httpx.AsyncClient(timeout=timeout, transport=_transport) as client:
        try:
            try:
                resp = await client.post(url, headers=headers, json=body)
            except httpx.TimeoutException:
                logger.info("Clef %s: timed out, retrying once", model)
                resp = await client.post(url, headers=headers, json=body)
            if resp.status_code == 429:
                try:
                    wait = float(resp.headers.get("retry-after", ""))
                except ValueError:
                    wait = None
                if wait is not None and wait <= MAX_RETRY_WAIT:
                    await asyncio.sleep(wait)
                    resp = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as e:
            logger.warning("Clef %s: request failed (%s)", model, type(e).__name__)
            return None

    if resp.status_code != 200:
        logger.warning("Clef %s: HTTP %d %s", model, resp.status_code, resp.text[:300])
        return None
    data = resp.json()
    result = data.get("result", data)  # the REST API wraps responses in {"result": ..., "success": ...}
    answers = result.get("answers") if isinstance(result, dict) else None
    if not isinstance(answers, dict):
        logger.warning("Clef %s: response without answers", model)
        return None
    usage = result.get("usage") or {}
    return ClefResult(answers, result.get("model", model), int(usage.get("input_tokens") or 0))
