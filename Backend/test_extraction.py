"""Tests for extraction.py: grounding, voting, safety net, review, fallback."""

import asyncio
import itertools

import pytest

import extraction
from extraction import grounded

TEXT = "Третий день температура 39, сильный кашель и болит горло. Насморка нет. Мне 34 года, женщина."
NO_REVIEW = {"emergency": False, "emergency_quote": "", "reason": "", "in_scope": True, "likely_condition": None}


def sample(findings, pain=None, travel=None, age=None, sex=None):
    return {"findings": [{"code": c, "status": s, "quote": q} for c, s, q in findings],
            "pain": pain, "travel": travel, "age": age, "sex": sex}


BASE = [("E_91", "present", "температура 39"), ("E_201", "present", "сильный кашель"),
        ("E_181", "absent", "Насморка нет")]


@pytest.fixture
def voting(monkeypatch):
    """Three extraction samples, as on a paid Groq tier (default is 1)."""
    monkeypatch.setattr(extraction, "EXTRACTION_SAMPLES", 3)


def run(fake_groq, samples, text=TEXT, review=NO_REVIEW):
    replies = itertools.cycle(samples)
    fake_groq.replies["extract"] = lambda _payload: next(replies)
    fake_groq.replies["review"] = review
    return asyncio.run(extraction.extract(text))


# ─── grounding ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("quote,expected", [
    ("температура 39", True),
    ("ТЕМПЕРАТУРА 39", True),                # case
    ("температура 39,", True),               # punctuation
    ("кашель сильный", True),                # reordered words
    ("высокая температура", False),          # paraphrase
    ("chest pain", False),                   # translation
    ("", False),
])
def test_grounded(quote, expected):
    assert grounded(quote, TEXT) is expected


def test_grounded_normalises_yo():
    assert grounded("ещё болит", "Ещё болит живот")
    assert grounded("еще болит", "Ещё болит живот")


# ─── extraction ──────────────────────────────────────────────────────────────

def test_present_absent_and_demographics(fake_groq):
    result = run(fake_groq, [sample(BASE, age=34, sex="F")])
    assert set(result.codes("present")) == {"E_91", "E_201"}
    assert result.codes("absent") == ["E_181"]
    assert (result.age, result.sex) == (34, "F")
    assert result.mode == "llm" and result.samples == extraction.EXTRACTION_SAMPLES == 1


def test_invalid_code_is_dropped(fake_groq):
    # codes are free strings in the schema; anything not a model feature is discarded
    result = run(fake_groq, [sample(BASE + [("E_9999", "present", "кашель")])])
    assert "E_9999" not in [f.code for f in result.findings]


def test_ungrounded_finding_is_dropped(fake_groq):
    invented = BASE + [("E_14", "present", "боль в груди в покое")]  # not in the text
    result = run(fake_groq, [sample(invented)])
    assert "E_14" not in result.codes("present") + result.codes("uncertain")


def test_minority_finding_becomes_uncertain(fake_groq, voting):
    with_extra = BASE + [("E_97", "present", "болит горло")]
    result = run(fake_groq, [sample(with_extra), sample(BASE), sample(BASE)])
    assert "E_97" in result.codes("uncertain")
    result = run(fake_groq, [sample(with_extra), sample(with_extra), sample(BASE)])
    assert "E_97" in result.codes("present")


def test_pain_block_maps_to_tokens(fake_groq):
    pain = {"locations": ["V_148"], "radiates_to": [], "character": [], "intensity": 6,
            "onset_speed": None, "quote": "болит горло"}
    result = run(fake_groq, [sample(BASE, pain=pain)])
    assert {"E_53", "E_55_@_V_148", "E_56_@_6"} <= set(result.codes("present"))


def test_ungrounded_pain_block_is_dropped(fake_groq):
    pain = {"locations": ["V_101"], "radiates_to": [], "character": [], "intensity": None,
            "onset_speed": None, "quote": "давит в груди"}
    result = run(fake_groq, [sample(BASE, pain=pain)])
    assert "E_53" not in result.codes("present")


def test_safety_net_adds_missed_red_flag(fake_groq):
    text = "Сильная боль в груди и потею"
    result = run(fake_groq, [sample([("E_50", "present", "потею")])], text=text)
    sources = {f.code: f.source for f in result.findings}
    assert sources.get("E_14") == "safety_net"


def test_safety_net_respects_grounded_denial(fake_groq):
    text = "Кашель, боли в груди нет"
    result = run(fake_groq, [sample([("E_201", "present", "Кашель"), ("E_14", "absent", "боли в груди нет")])],
                 text=text)
    assert "E_14" in result.codes("absent")
    assert "E_14" not in result.codes("present")


def test_review_emergency_must_be_quoted(fake_groq):
    text = "Внезапно стало трудно дышать, губы синеют"
    verified = {**NO_REVIEW, "emergency": True, "emergency_quote": "губы синеют"}
    result = run(fake_groq, [sample([])], text=text, review=verified)
    assert result.review["emergency"] and result.review["emergency_verified"]

    invented = {**NO_REVIEW, "emergency": True, "emergency_quote": "потерял сознание"}
    result = run(fake_groq, [sample([])], text=text, review=invented)
    assert result.review["emergency"] and not result.review["emergency_verified"]


def test_review_out_of_scope(fake_groq):
    review = {**NO_REVIEW, "in_scope": False, "likely_condition": "Appendicitis"}
    result = run(fake_groq, [sample(BASE)], review=review)
    assert result.review["out_of_scope"] and result.review["likely_condition"] == "Appendicitis"


def test_falls_back_to_regex_without_llm(fake_groq):
    fake_groq.replies["extract"] = 500
    fake_groq.replies["review"] = 500
    result = asyncio.run(extraction.extract(TEXT))
    assert result.mode == "fallback"
    assert {"E_91", "E_201"} <= set(result.codes("present"))
    assert result.review is None


def test_catalog_codes_are_model_features():
    from ml_model import _feat_idx
    assert all(code in _feat_idx for code in extraction.BINARY_CODES)
    assert all(f"E_55_@_{v}" in _feat_idx for v in extraction.COMMON_LOCATIONS)
    assert all(f"E_54_@_{v}" in _feat_idx for v in extraction._PAIN_CHARACTER)
    assert len(extraction.COMMON_LOCATIONS) == 70


def test_short_label():
    assert extraction.short_label("E_91") == "A fever (either felt or measured with a thermometer)"
    assert extraction.short_label("E_14") == "Chest pain even at rest"


def test_evidence_label():
    assert extraction.evidence_label("E_91")["ru"].startswith("У вас есть температура")
    assert extraction.evidence_label("E_55_@_V_148")["en"] == "Pain location: pharynx"


# ─── Backend switch (EXTRACTION_BACKEND) ─────────────────────────────────────

def test_default_backend_is_groq(fake_groq, fake_clef, monkeypatch):
    monkeypatch.delenv("EXTRACTION_BACKEND", raising=False)
    fake_groq.replies.update({"extract": sample(BASE), "review": NO_REVIEW})
    result = asyncio.run(extraction.extract_complaint(TEXT))
    assert result.mode == "llm" and fake_clef["requests"] == []


def test_clef_backend(fake_groq, fake_clef, monkeypatch):
    monkeypatch.setenv("EXTRACTION_BACKEND", "clef")
    from conftest import status
    fake_clef["answers"]["E_91"] = status(reported=0.95)
    result = asyncio.run(extraction.extract_complaint(TEXT))
    assert result.mode == "clef" and "E_91" in result.codes("present")
    assert fake_groq.calls == []


def test_clef_failure_falls_back_to_groq(fake_groq, fake_clef, monkeypatch):
    monkeypatch.setenv("EXTRACTION_BACKEND", "clef")
    fake_clef["fail"] = True
    fake_groq.replies.update({"extract": sample(BASE), "review": NO_REVIEW})
    result = asyncio.run(extraction.extract_complaint(TEXT))
    assert result.mode == "llm"
    assert set(result.codes("present")) == {"E_91", "E_201"}
