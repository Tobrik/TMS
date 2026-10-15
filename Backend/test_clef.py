"""Tests for clef.py and extraction_clef.py against a fake Workers AI API."""

import asyncio

import clef
import extraction_clef as xc
from conftest import choice, status

TEXT = "Третий день температура 39, сильный кашель, болит горло. Насморка нет."


def test_request_shape(fake_clef):
    result = asyncio.run(clef.ask("text", {"q": {"type": "noul", "instructions": "?"}}))
    path, body = fake_clef["requests"][0]
    assert path == "/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef-flash"
    assert body == {"model": "clef-flash", "state": "text", "questions": {"q": {"type": "noul", "instructions": "?"}}}
    assert result.input_tokens == 2000


def test_batches_respect_question_limit():
    assert all(len(b) <= clef.MAX_QUESTIONS for b in xc.SYMPTOM_BATCHES)
    assert len(xc.CONTEXT_QUESTIONS) <= clef.MAX_QUESTIONS
    assert sum(len(b) for b in xc.SYMPTOM_BATCHES) == len(xc.BINARY_CODES)


def test_confidence_bands(fake_clef):
    fake_clef["answers"].update({
        "E_91": status(reported=0.9),               # sure → pre-ticked
        "E_201": status(reported=0.6, denied=0.1),  # unsure → "?" chip
        "E_97": status(reported=0.3),               # unsure → "?" chip
        "E_14": status(reported=0.1),               # ignored
        "E_181": status(denied=0.85),               # sure denial → pre-marked "no"
        "E_94": status(denied=0.6),                 # unsure denial → left out
    })
    result = asyncio.run(xc.extract(TEXT))
    assert result.mode == "clef"
    assert "E_91" in result.codes("present")
    assert {"E_201", "E_97"} <= set(result.codes("uncertain"))
    assert result.codes("absent") == ["E_181"]
    assert "E_14" not in result.codes("present") + result.codes("uncertain")
    assert "E_94" not in result.codes("absent")
    confidence = {f.code: f.confidence for f in result.findings}
    assert confidence["E_91"] == 0.9 and confidence["E_201"] == 0.6
    assert result.usage["input_tokens"] == 2000 * 5


def test_pain_location_sets_pain_to_the_same_confidence(fake_clef):
    fake_clef["answers"]["pain_location"] = choice({"none": 0.1, "V_148": 0.85, "V_89": 0.05})
    fake_clef["answers"]["pain_intensity"] = choice({"not_stated": 0.3, "7": 0.6, "6": 0.1})
    result = asyncio.run(xc.extract(TEXT))
    assert {"E_53", "E_55_@_V_148"} <= set(result.codes("present"))
    assert "E_56_@_7" in result.codes("uncertain")

    fake_clef["answers"]["pain_location"] = choice({"none": 0.4, "V_148": 0.6})
    result = asyncio.run(xc.extract(TEXT))
    assert {"E_53", "E_55_@_V_148"} <= set(result.codes("uncertain"))


def test_emergency_is_only_a_soft_warning(fake_clef):
    fake_clef["answers"]["emergency"] = {"type": "noul", "noul": 0.85}
    assert not asyncio.run(xc.extract(TEXT)).review["emergency"]
    fake_clef["answers"]["emergency"] = {"type": "noul", "noul": 0.95}
    review = asyncio.run(xc.extract(TEXT)).review
    assert review["emergency"] and not review["emergency_verified"]
    assert review["emergency_probability"] == 0.95


def test_out_of_scope(fake_clef):
    fake_clef["answers"]["scope"] = choice({"in_scope": 0.2, "appendicitis": 0.7, "other_outside": 0.1})
    review = asyncio.run(xc.extract(TEXT)).review
    assert review["out_of_scope"] and review["likely_condition"] == "appendicitis"


def test_interpret_accepts_saved_raw_output(fake_clef):
    fake_clef["answers"]["E_91"] = status(reported=0.9)
    first = asyncio.run(xc.extract(TEXT))
    again = xc.interpret(first.raw, TEXT, first.model)
    assert again.codes("present") == first.codes("present")
    assert again.review == first.review


def test_no_travel_default_is_not_a_finding(fake_clef):
    fake_clef["answers"]["travel"] = choice({"not_mentioned": 0.2, "V_10": 0.8})
    assert not any(c.startswith("E_204") for c in asyncio.run(xc.extract(TEXT)).codes("present"))
    fake_clef["answers"]["travel"] = choice({"not_mentioned": 0.1, "V_10": 0.1, "V_6": 0.8})
    assert "E_204_@_V_6" in asyncio.run(xc.extract(TEXT)).codes("present")


def test_raw_probabilities_kept(fake_clef):
    fake_clef["answers"]["E_91"] = status(reported=0.9)
    raw = asyncio.run(xc.extract(TEXT)).raw
    assert raw["E_91"]["reported"] == 0.9
    assert raw["emergency"] == {"yes": 0.02}


def test_any_failed_request_falls_back(fake_clef):
    fake_clef["fail"] = True
    assert asyncio.run(xc.extract(TEXT)).mode == "fallback"


def test_no_credentials_falls_back(monkeypatch):
    monkeypatch.delenv("CF_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CF_API_TOKEN", raising=False)
    assert asyncio.run(xc.extract(TEXT)).mode == "fallback"
